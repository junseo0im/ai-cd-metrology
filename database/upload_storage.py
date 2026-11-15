"""Local byte-preserving upload storage; cleanup is limited to owned new files.

No recursive delete, overwrite, image decoding, resizing, or source mutation.
Synchronous errors are compensated; process-crash orphan recovery is a later
operational concern, not a distributed filesystem/SQLite transaction.
"""
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid
import io
from pathlib import PurePosixPath
import stat
import struct
import zipfile
import zlib


def safe_filename(name, extensions=(".png",)):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or name[-1:] in (".", " ") or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.lower().split(".")[0] in
            {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1,10)],
             *[f"lpt{i}" for i in range(1,10)]}):
        raise ValueError("Unsafe upload filename; a PNG basename without path characters is required")
    if Path(name).suffix.lower() not in extensions:
        raise ValueError("Unsupported upload format")
    return name


class NewFiles:
    """Own a UUID directory and exclusive-created files; preserve existing paths."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.directory = None
        self.files = []
        self.directories = []
        self.kept = False

    def _allocate(self, filename):
        safe_filename(filename)
        if self.directory is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.directory = self.root / uuid.uuid4().hex
            self.directory.mkdir()  # Never adopt a pre-existing directory.
            self.directories.append(self.directory)
        folder = self.directory / uuid.uuid4().hex
        folder.mkdir()
        self.directories.append(folder)
        return folder / filename

    def write(self, filename, writer):
        path = self._allocate(filename)
        with path.open("xb") as target:
            info = os.fstat(target.fileno())
            self.files.append((path, info.st_dev, info.st_ino))
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        return path

    def copy(self, source, filename, expected_sha256):
        def writer(target):
            with Path(source).open("rb") as stream:
                shutil.copyfileobj(stream, target)
        target = self.write(filename, writer)
        if file_sha256(target) != expected_sha256:
            raise ValueError("Stored image hash differs from the approved preview")
        return target

    def preserve(self):
        self.kept = True

    def cleanup(self):
        if self.kept:
            return
        errors = []
        for path, device, inode in reversed(self.files):
            try:
                if not path.exists() and not path.is_symlink():
                    continue
                info = path.lstat()
                if (path.is_symlink() or (info.st_dev, info.st_ino) != (device, inode)
                        or not path.resolve().is_relative_to(self.root)):
                    raise RuntimeError(f"Cleanup refused an unowned/replaced path: {path}")
                path.unlink()
            except OSError as exc:
                errors.append(str(exc))
            except RuntimeError as exc:
                errors.append(str(exc))
        for folder in reversed(self.directories):
            try:
                folder.rmdir()  # Non-empty/unexpected content is never recursively removed.
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Upload cleanup needs review: " + "; ".join(errors))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_signature(uploads):
    """Identity includes content, order and original name; never filename alone."""
    digest = hashlib.sha256()
    for upload in uploads:
        name = upload["filename"].encode("utf-8")
        content = upload["content"]
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


MAX_IMAGES = 512
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2048
MAX_COMPRESSION_RATIO = 200
MAX_DECODED_PNG_BYTES = 256 * 1024 * 1024


def validate_png_bytes(data):
    """Bounded PNG framing/CRC and zlib validation; no re-encoding or resizing."""
    if data[:8]!=bytes((137,80,78,71,13,10,26,10)):
        raise ValueError("깨진 PNG signature")
    offset=8
    saw_header=saw_data=saw_end=False
    decoder=zlib.decompressobj()
    decoded=0
    while offset<len(data):
        if offset+12>len(data):
            raise ValueError("잘린 PNG chunk")
        size=struct.unpack(">I",data[offset:offset+4])[0]
        end=offset+12+size
        if end>len(data):
            raise ValueError("잘린 PNG chunk")
        kind=data[offset+4:offset+8]
        payload=data[offset+8:offset+8+size]
        crc=struct.unpack(">I",data[offset+8+size:end])[0]
        if zlib.crc32(kind+payload)&0xffffffff != crc:
            raise ValueError("PNG CRC 오류")
        if not saw_header:
            if kind!=b"IHDR" or size!=13 or min(struct.unpack(">II",payload[:8]))<=0:
                raise ValueError("Invalid PNG IHDR")
            saw_header=True
        elif kind==b"IHDR":
            raise ValueError("Duplicate PNG IHDR")
        if kind==b"IDAT":
            saw_data=True
            pending=payload
            try:
                while pending:
                    output=decoder.decompress(pending,65536)
                    decoded+=len(output)
                    if decoded>MAX_DECODED_PNG_BYTES:
                        raise ValueError("PNG decoded size limit exceeded")
                    pending=decoder.unconsumed_tail
            except zlib.error as exc:
                raise ValueError("깨진 PNG compressed data") from exc
        if kind==b"IEND":
            if size or end!=len(data):
                raise ValueError("Invalid PNG IEND")
            saw_end=True
        offset=end
    if not (saw_header and saw_data and saw_end and decoder.eof) or decoder.unused_data:
        raise ValueError("불완전한 PNG")



def _check_zip_directory(content):
    # Bound ZipInfo allocation before opening an untrusted central directory.
    marker=b"PK"+bytes((5,6))
    start=max(0,len(content)-65557)
    position=content.rfind(marker,start)
    while position>=start:
        if position+22<=len(content):
            values=struct.unpack_from("<4s4H2IH",content,position)
            if position+22+values[-1]==len(content):
                break
        position=content.rfind(marker,start,position)
    else:
        raise ValueError("깨진 ZIP directory")
    _,disk,cd_disk,disk_entries,entries,size,offset,_=values
    if disk or cd_disk or entries!=disk_entries or entries==65535 or offset==0xffffffff:
        raise ValueError("분할 ZIP/ZIP64는 지원하지 않습니다")
    if entries>MAX_ARCHIVE_ENTRIES or offset+size>position:
        raise ValueError("ZIP directory 개수/범위 오류")
    end=offset+size
    count=0
    while offset<end:
        if offset+46>end or content[offset:offset+4]!=b"PK"+bytes((1,2)):
            raise ValueError("깨진 ZIP directory entry")
        name,extra,comment=struct.unpack_from("<3H",content,offset+28)
        offset+=46+name+extra+comment
        count+=1
        if count>MAX_ARCHIVE_ENTRIES or offset>end:
            raise ValueError("ZIP 항목 개수/범위 제한 초과")
    if count!=entries:
        raise ValueError("ZIP directory count mismatch")

def expand_uploads(uploads):
    """Validate ZIP members in memory, then reuse native PNG staging. No extractall."""
    expanded=[]
    total=0
    def add(name, content, **origin):
        nonlocal total
        safe_filename(name)
        if len(content)>MAX_FILE_BYTES:
            raise ValueError("PNG 파일별 용량 제한 초과")
        total+=len(content)
        if total>MAX_TOTAL_BYTES or len(expanded)>=MAX_IMAGES:
            raise ValueError("전체 이미지 개수/용량 제한 초과")
        validate_png_bytes(content)
        expanded.append({"filename":name,"content":content,**origin})
    for upload in uploads:
        name=safe_filename(upload["filename"],(".png",".zip"))
        content=upload["content"]
        if not isinstance(content,bytes):
            raise ValueError("Upload content must be bytes")
        if name.lower().endswith(".png"):
            add(name,content)
            continue
        if len(content)>MAX_ARCHIVE_BYTES:
            raise ValueError("ZIP 업로드 용량 제한 초과")
        _check_zip_directory(content)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries=archive.infolist()
                if len(entries)>MAX_ARCHIVE_ENTRIES:
                    raise ValueError("ZIP 항목 개수 제한 초과")
                seen=set()
                declared=0
                for info in entries:
                    member=info.orig_filename
                    parts=member.rstrip("/").split("/")
                    if ("\\" in member or ":" in member or PurePosixPath(member).is_absolute()
                            or any(part in ("",".","..") for part in parts)):
                        raise ValueError("Unsafe ZIP member path")
                    mode=(info.external_attr>>16)&0xffff
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
                        raise ValueError("ZIP symbolic/special links are forbidden")
                    if info.flag_bits&1:
                        raise ValueError("Encrypted ZIP is not supported")
                    key=member.casefold()
                    if key in seen:
                        raise ValueError("ZIP 내부 중복 경로")
                    seen.add(key)
                    if info.is_dir():
                        continue
                    safe_filename(parts[-1])
                    declared+=info.file_size
                    if (info.file_size>MAX_FILE_BYTES or info.file_size+total>MAX_TOTAL_BYTES
                            or info.file_size/max(info.compress_size,1)>MAX_COMPRESSION_RATIO):
                        raise ValueError("ZIP 크기/압축 비율 제한 초과")
                    with archive.open(info) as stream:
                        data=stream.read(MAX_FILE_BYTES+1)
                    if len(data)!=info.file_size:
                        raise ValueError("ZIP member size mismatch")
                    add(parts[-1],data,source_archive=name,archive_member=member)
        except (zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:
            raise ValueError("ZIP 파일을 읽을 수 없습니다") from exc
    if uploads and not expanded:
        raise ValueError("등록할 PNG가 없는 ZIP입니다")
    return expanded


def stage_uploads(uploads, staging_root):
    """Return staged input dictionaries and their owned-file cleanup handle."""
    original_signature = upload_signature(uploads)
    uploads = expand_uploads(uploads)
    files = NewFiles(staging_root)
    inputs = []
    try:
        for upload in uploads:
            name = safe_filename(upload["filename"])
            content = upload["content"]
            if not isinstance(content, bytes):
                raise ValueError("Upload content must be original bytes")
            path = files.write(name, lambda target, content=content: target.write(content))
            inputs.append({"image_id": "IMG-" + uuid.uuid4().hex, "image_path": str(path),
                           **{key:upload[key] for key in ("source_archive","archive_member") if key in upload}})
        return {"signature": original_signature, "inputs": inputs, "files": files}
    except Exception:
        files.cleanup()
        raise
