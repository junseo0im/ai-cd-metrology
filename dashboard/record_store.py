"""Local record JSON persistence only. No SQLite, calibration, AI or Runner imports."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile

from dashboard.record_defaults import definition,STATUSES
from dashboard import auth

PROJECT_ROOT=Path(__file__).resolve().parents[1]
ROW_FIELDS={"id","item","value","unit","status","note"}


class RecordError(ValueError):
    pass


class RecordConflict(RecordError):
    pass


def editing_enabled():
    """Server-validated ADMIN grant; the legacy environment toggle is ignored."""
    return auth.is_admin()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def tab_token(tab):
    return hashlib.sha256(json.dumps(tab,ensure_ascii=False,sort_keys=True,allow_nan=False).encode("utf-8")).hexdigest()


def default_rows(document_id,tab_id):
    tabs=definition(document_id)["tabs"]
    spec=next((rows for key,title,rows in tabs if key==tab_id),None)
    if spec is None:
        raise RecordError("알 수 없는 기록 탭입니다.")
    return [{key:deepcopy(value) for key,value in row.items() if key in ROW_FIELDS} for row in spec]


def validate_rows(document_id,tab_id,rows):
    tabs=definition(document_id)["tabs"]
    specs=next((items for key,title,items in tabs if key==tab_id),None)
    if specs is None or not isinstance(rows,list):
        raise RecordError("표 형식 또는 탭 정보가 올바르지 않습니다.")
    if len(rows)!=len(specs) or any(not isinstance(row,dict) or set(row)!=ROW_FIELDS for row in rows):
        raise RecordError("표의 항목 수 또는 열이 바뀌었습니다. 취소 후 다시 수정하세요.")
    by_id={row["id"]:row for row in rows if isinstance(row["id"],str)}
    if len(by_id)!=len(specs) or set(by_id)!={row["id"] for row in specs}:
        raise RecordError("표의 항목 ID가 올바르지 않습니다.")
    result=[]
    for spec in specs:
        row=by_id[spec["id"]]
        if any(not isinstance(value,str) for value in row.values()) or row["item"]!=spec["item"]:
            raise RecordError("항목 이름은 변경할 수 없으며 각 셀은 문자열이어야 합니다.")
        if any(len(row[key])>4000 for key in ("value","note")) or len(row["unit"])>40:
            raise RecordError(spec["item"]+": 기록값/비고는 4000자, 단위는 40자 이내로 입력하세요.")
        cleaned={key:value.strip() for key,value in row.items()}
        value=cleaned["value"];status=cleaned["status"]
        if status not in STATUSES:
            raise RecordError(spec["item"]+": 상태는 확정 / 검토 필요 / 미입력 중 선택하세요.")
        if not value and status=="확정":
            raise RecordError(spec["item"]+": 값이 없으면 미입력 또는 검토 필요를 선택하세요.")
        if value and status=="미입력":
            raise RecordError(spec["item"]+": 값이 있으므로 확정 또는 검토 필요를 선택하세요.")
        if value and spec["kind"]!="text":
            try:
                number=Decimal(value)
            except InvalidOperation as exc:
                raise RecordError(spec["item"]+": 숫자만 입력하세요. 단위는 단위 열에 적습니다.") from exc
            if not number.is_finite():
                raise RecordError(spec["item"]+": NaN / Infinity는 입력할 수 없습니다.")
            if spec["kind"]=="positive" and number<=0:
                raise RecordError(spec["item"]+": 0보다 큰 수를 입력하세요.")
            if spec["kind"] in {"nonnegative","count"} and number<0:
                raise RecordError(spec["item"]+": 음수는 입력할 수 없습니다.")
            if spec["kind"]=="count" and number!=number.to_integral_value():
                raise RecordError(spec["item"]+": 정수를 입력하세요.")
        result.append(cleaned)
    return result


def _timestamp(value):
    try:
        parsed=datetime.fromisoformat(value)
    except (ValueError,TypeError) as exc:
        raise RecordError("수정 시각의 ISO 8601 형식이 올바르지 않습니다.") from exc
    if parsed.tzinfo is None:
        raise RecordError("수정 시각에는 시간대가 필요합니다.")


def validate_document(document_id,doc):
    spec=definition(document_id)
    if not isinstance(doc,dict) or set(doc)!={"schema_version","document_id","record_version","created_at","updated_at","tabs"}:
        raise RecordError("JSON 기록 구조가 올바르지 않습니다. 기존 파일을 보존합니다.")
    if doc["schema_version"]!=1 or type(doc["schema_version"]) is not int or doc["document_id"]!=document_id:
        raise RecordError("지원하지 않는 기록 버전 또는 문서 종류입니다. 자동 변환하지 않습니다.")
    if type(doc["record_version"]) is not int or doc["record_version"]<1:
        raise RecordError("기록 버전이 올바르지 않습니다.")
    _timestamp(doc["created_at"]);_timestamp(doc["updated_at"])
    if not isinstance(doc["tabs"],dict) or set(doc["tabs"])!={key for key,title,rows in spec["tabs"]}:
        raise RecordError("탭 구성이 다릅니다. 기존 파일을 보존합니다.")
    for key,title,rows in spec["tabs"]:
        tab=doc["tabs"][key]
        if not isinstance(tab,dict) or set(tab)!={"title","revision","updated_at","rows"} or tab["title"]!=title:
            raise RecordError("탭의 JSON 구조가 올바르지 않습니다.")
        if type(tab["revision"]) is not int or tab["revision"]<1:
            raise RecordError("탭 기록 버전이 올바르지 않습니다.")
        _timestamp(tab["updated_at"])
        validate_rows(document_id,key,tab["rows"])
    return doc


def new_document(document_id):
    stamp=now_iso()
    spec=definition(document_id)
    return {"schema_version":1,"document_id":document_id,"record_version":1,
            "created_at":stamp,"updated_at":stamp,
            "tabs":{key:{"title":title,"revision":1,"updated_at":stamp,
                         "rows":default_rows(document_id,key)} for key,title,rows in spec["tabs"]}}


@contextmanager
def file_lock(path):
    lock=path.with_name(path.name+".lock")
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RecordConflict("다른 저장 작업이 진행 중입니다. 잠시 후 다시 시도하세요. 비정상 종료 뒤 잠금이 남으면 관리자 확인이 필요합니다.") from exc
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def atomic_write(path,doc):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                prefix="."+path.name+".",suffix=".tmp",delete=False,newline="\n") as handle:
            temporary=Path(handle.name)
            json.dump(doc,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Never delete or truncate the existing normal file.


class RecordStore:
    def __init__(self,directory=None):
        self.directory=Path(directory if directory is not None else
                            os.environ.get("IDA_RECORDS_DIR",PROJECT_ROOT/"runtime_data"/"config")).resolve()

    def path(self,document_id):
        definition(document_id)  # Only these two fixed filenames are permitted.
        return self.directory/(document_id+".json")

    def _read(self,document_id,path):
        try:
            doc=json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError,json.JSONDecodeError) as exc:
            raise RecordError("기존 JSON을 읽을 수 없습니다. 원본 파일은 덮어쓰지 않습니다. 관리자에게 복구를 요청하세요.") from exc
        return validate_document(document_id,doc)

    def load(self,document_id):
        path=self.path(document_id)
        if not path.exists():
            self.directory.mkdir(parents=True,exist_ok=True)
            with file_lock(path):
                if not path.exists():
                    doc=new_document(document_id);validate_document(document_id,doc)
                    atomic_write(path,doc)
        return self._read(document_id,path)

    def save_tab(self,document_id,tab_id,rows,*,expected_token):
        if not editing_enabled():
            raise PermissionError("기록 변경에는 ADMIN 로그인이 필요합니다.")
        cleaned=validate_rows(document_id,tab_id,rows)
        path=self.path(document_id)
        with file_lock(path):
            current=self._read(document_id,path)
            tab=current["tabs"][tab_id]
            if expected_token!=tab_token(tab):
                raise RecordConflict("이 탭을 다른 세션 또는 파일 편집으로 변경했습니다. 취소 후 최신 내용을 다시 불러오세요.")
            if not editing_enabled():
                raise PermissionError("기록 변경에는 ADMIN 로그인이 필요합니다.")
            stamp=now_iso()
            tab["rows"]=cleaned;tab["revision"]+=1;tab["updated_at"]=stamp
            current["record_version"]+=1;current["updated_at"]=stamp
            validate_document(document_id,current)
            atomic_write(path,current)
        return current
