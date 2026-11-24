"""전체 Dataset deletion gate (TASK 18-B-12A). Every check runs again at execution time on the server.

* Only an authenticated ADMIN (dashboard.auth server-side grant) may delete; GUEST can search/view only.
* The operational DB additionally needs the explicit server switch IDA_DATASET_DELETE_ENABLED=1
  (unset by default -> operational deletion disabled). Temporary/test DBs need ADMIN only.
* No deletion while an analysis runs (its images could be deleted underneath the Runner).
* The selection must still be a subset of the CURRENT filtered result (re-queried here).
"""
import os

from dashboard import auth
from database.image_deletion import DeletionRefused, delete_images, plan_deletion
from database.migrate_v4 import refuse_operational

DELETE_SWITCH = "IDA_DATASET_DELETE_ENABLED"
WARNING = ("선택한 원본 이미지와 종속 분석 데이터가 영구 삭제됩니다. "
           "이 작업은 실행 후 일반적인 화면 조작으로 복구할 수 없습니다.")
CONFIRM_PHRASE = "영구 삭제"


def _operational(provider):
    try:
        refuse_operational(provider.db_path)
        return False
    except PermissionError:
        return True


def _analysis_running():
    from dashboard.components.analysis_management import job_snapshot
    return job_snapshot() is not None


def deletion_status(provider):
    """(enabled, reason). Reason is shown instead of an active delete button."""
    if not auth.is_admin():
        return False, "영구 삭제는 관리자 전용입니다. GUEST는 검색·조회만 할 수 있습니다."
    if _operational(provider) and os.environ.get(DELETE_SWITCH) != "1":
        return False, "운영 DB 영구 삭제는 서버 설정으로 비활성 상태입니다 (관리자 승인 후 활성화)."
    if _analysis_running():
        return False, "분석 실행 중에는 삭제할 수 없습니다. 완료 후 다시 시도하세요."
    return True, ""


def _validated(provider, image_ids, filters):
    ids = sorted(set(image_ids or ()))
    if not ids:
        raise DeletionRefused("삭제할 이미지를 선택하세요.")
    current = set(provider.dataset_ids(filters))
    if not set(ids) <= current:
        raise DeletionRefused("선택 이미지가 현재 검색 조건과 일치하지 않습니다. 다시 검색하세요.")
    return ids


def result_root():
    from dashboard.components.analysis_management import analysis_output_root
    return analysis_output_root()


def prepare(provider, image_ids, filters):
    enabled, reason = deletion_status(provider)
    if not enabled:
        raise DeletionRefused(reason)
    ids = _validated(provider, image_ids, filters)
    return plan_deletion(provider.db_path, ids, storage_root=provider.storage_root, result_root=result_root())


def execute(provider, image_ids, filters, token, phrase):
    enabled, reason = deletion_status(provider)          # Re-checked now, not taken from the page.
    if not enabled:
        raise DeletionRefused(reason)
    if phrase != CONFIRM_PHRASE:
        raise DeletionRefused("확인 문구가 일치하지 않습니다.")
    ids = _validated(provider, image_ids, filters)
    return delete_images(provider.db_path, ids, expected_token=token, storage_root=provider.storage_root,
                         result_root=result_root(), actor=auth.current_user()["display_name"],
                         allow_operational=_operational(provider) and os.environ.get(DELETE_SWITCH) == "1")
