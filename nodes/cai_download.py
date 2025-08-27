# -*- coding: utf-8 -*-

# =========================================================
# Kw_Json_Lora_CivitAIDownloader
# - 以 modelVersionId 为唯一键
# - 加载顺序：本地 -> R2 -> Civitai
# - 成功后回传 R2（本地命中但 R2 缺失也回灌）
# - R2 统一相对键（= modelVersionId），list 去全局前缀
# - R2 元数据 ASCII 化（避免 Non-ascii metadata 报错）
# - 兼容缺失 requests/boto3 的自动安装（可关闭）
# - 兼容旧代码输入；新增 R2 输入项为可选
# =========================================================

import importlib
import subprocess
import sys
import logging
import os
import re
import json
import threading
from typing import Optional, Tuple, Dict

# ---------------- logging ----------------
_LOG_LEVEL = os.environ.get("KW_DL_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("KwCivitAI")

# ---------------- auto-install helper ----------------
def _ensure_pkg(pkg_name: str, auto_install: bool, logger):
    try:
        importlib.import_module(pkg_name)
        return True
    except ImportError:
        if not auto_install:
            logger.warning(f"[deps] Missing '{pkg_name}' and auto_install=OFF.")
            return False
        try:
            logger.info(f"[deps] Installing '{pkg_name}' ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.invalidate_caches()
            importlib.import_module(pkg_name)
            logger.info(f"[deps] Installed '{pkg_name}'.")
            return True
        except Exception as e:
            logger.error(f"[deps] Failed to install '{pkg_name}': {e}")
            return False

# requests 占位（后面按 auto_install 再确保）
try:
    import requests
    from requests.exceptions import HTTPError
except Exception:
    requests = None
    HTTPError = Exception

# ComfyUI 环境
import folder_paths

# ---------------- models 根路径 ----------------
def get_base_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    models_dir = os.path.join(base_dir, 'models')
    return models_dir

def get_model_dirs():
    models_dir = get_base_dir()
    if not os.path.isdir(models_dir):
        return []
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _normpath(p: str) -> str:
    return os.path.normpath(p)

# =========================================================
# R2 Client (S3 compatible)
# - 相对键：modelVersionId
# - list 去掉全局前缀
# - ASCII 元数据
# =========================================================
class R2Client:
    def __init__(
        self,
        enabled: bool,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        auto_install: bool = False,
    ):
        self.enabled = enabled and all([endpoint, access_key, secret_key, bucket])
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.prefix = (prefix or "").lstrip("/").rstrip("/")
        self.region = region
        self.auto_install = auto_install
        self._boto3 = None
        self._client = None
        if self.enabled:
            self._init_client()

    def _init_client(self):
        if not _ensure_pkg("boto3", self.auto_install, logger):
            logger.warning("[r2] boto3 not available; R2 disabled.")
            self.enabled = False
            return
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
        self._boto3 = boto3
        # Cloudflare R2 支持 virtual-hosted 风格
        cfg = Config(s3={"addressing_style": "virtual"})
        try:
            session = boto3.session.Session()
            self._client = session.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=cfg,
            )
            logger.info(f"[r2] Initialized R2 client. bucket={self.bucket} prefix='{self.prefix}'")
            # 轻量凭证探测：失败不阻断流程，但提示
            try:
                self._client.list_objects_v2(Bucket=self.bucket, MaxKeys=1, Prefix=self._full_key(""))
                logger.info("[r2] Credential check OK.")
            except Exception as e:
                msg = str(e)
                logger.warning(f"[r2] Credential check warning: {msg}")
                if "Credential access key has length" in msg or "The AWS Access Key Id" in msg or "SignatureDoesNotMatch" in msg:
                    logger.warning("[r2] ⚠️ 请使用 R2 的 Access Key ID / Secret Access Key（不是 API Token）。")
        except Exception as e:
            logger.error(f"[r2] init failed: {e}")
            self.enabled = False

    def _full_key(self, relative_key: str) -> str:
        rk = (relative_key or "").lstrip("/")
        if not self.prefix:
            return rk
        return f"{self.prefix}/{rk}"

    def list(self, prefix: str = ""):
        if not self.enabled:
            return []
        real_prefix = self._full_key(prefix)
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=real_prefix):
                contents = page.get("Contents") or []
                for obj in contents:
                    k = obj["Key"]
                    if self.prefix and k.startswith(self.prefix + "/"):
                        k = k[len(self.prefix) + 1:]
                    keys.append(k)
            logger.info(f"[r2] list('{prefix}') -> {len(keys)} keys")
            return keys
        except Exception as e:
            logger.error(f"[r2] list error: {e}")
        return []

    def head(self, relative_key: str) -> bool:
        if not self.enabled:
            logger.info(f"[r2] disabled -> head MISS: {relative_key}")
            return False
        key = self._full_key(relative_key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            logger.info(f"[r2] head HIT: {relative_key}")
            return True
        except Exception as e:
            logger.info(f"[r2] head MISS: {relative_key} ({e.__class__.__name__})")
            return False

    def download(self, relative_key: str, local_path: str) -> bool:
        if not self.enabled:
            return False
        key = self._full_key(relative_key)
        _ensure_dir(os.path.dirname(local_path))
        try:
            logger.info(f"[r2] download s3://{self.bucket}/{key} -> {local_path}")
            with open(local_path, "wb") as f:
                self._client.download_fileobj(self.bucket, key, f)
            return True
        except Exception as e:
            logger.error(f"[r2] download failed: {relative_key} -> {e}")
            return False

    @staticmethod
    def _ascii_meta(meta: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not meta:
            return {}
        out = {}
        for k, v in meta.items():
            try:
                k2 = str(k).encode("ascii", "ignore").decode("ascii")
                v2 = str(v).encode("ascii", "ignore").decode("ascii")
                out[k2] = v2
            except Exception:
                pass
        return out

    def upload(self, relative_key: str, local_path: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self.enabled:
            return False
        key = self._full_key(relative_key)
        meta = self._ascii_meta(metadata)
        try:
            logger.info(f"[r2] upload {local_path} -> s3://{self.bucket}/{key} meta={meta}")
            with open(local_path, "rb") as f:
                self._client.upload_fileobj(
                    f, self.bucket, key,
                    ExtraArgs={"Metadata": meta} if meta else None
                )
            logger.info(f"[r2] upload OK: key={relative_key}")
            return True
        except Exception as e:
            msg = str(e)
            logger.error(f"[r2] upload failed: {relative_key} -> {msg}")
            if "Credential access key has length" in msg:
                logger.error("[r2] ❌ Access Key 长度异常：请到 R2 控制台创建 'Access Key ID / Secret Access Key'（不是 API Token）。")
            elif "SignatureDoesNotMatch" in msg or "The AWS Access Key Id" in msg:
                logger.error("[r2] ❌ 签名失败：请检查 endpoint、access_key、secret_key、bucket 是否正确。")
            return False

# =========================================================
# 网络下载（Civitai）
# =========================================================
def download_file_with_token(fname: str, url: str, params=None, save_path='.', logger=logger) -> bool:
    _ensure_dir(save_path)
    file_path = _normpath(os.path.join(save_path, fname))
    if os.path.exists(file_path):
        logger.info(f"[net] Exists, skip: {file_path}")
        return True

    global requests, HTTPError
    if requests is None:
        ok = _ensure_pkg("requests", True, logger)
        if not ok:
            logger.error("[net] requests not available.")
            return False
        import requests as _rq  # type: ignore
        from requests.exceptions import HTTPError as _HE  # type: ignore
        requests = _rq
        HTTPError = _HE

    try:
        with requests.get(url, params=params or {}, stream=True, timeout=60) as response:
            response.raise_for_status()
            logger.info(f"[net] GET {response.url}")
            total_size = int(response.headers.get('content-length', 0)) or None
            downloaded_size = 0
            progress_interval = 5 * 1024 * 1024  # 5MB
            next_progress = progress_interval
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size and downloaded_size >= next_progress:
                        pct = downloaded_size / total_size * 100
                        logger.info(f"[net] Download progress: {pct:.1f}% ({downloaded_size}/{total_size} bytes)")
                        next_progress += progress_interval
        logger.info(f"[net] Downloaded: {file_path}")
        return True
    except HTTPError as http_err:
        logger.error(f"[net] HTTP error: {http_err}")
    except Exception as err:
        logger.error(f"[net] Error: {err}")
    return False

def _civitai_download(local_fname: str, model_version_id: str, token: str, local_dir: str, full_url: Optional[str]) -> Tuple[bool, Optional[str]]:
    _ensure_dir(local_dir)
    url = full_url.strip() if full_url else f'https://civitai.com/api/download/models/{model_version_id}'
    params = {'token': token} if token else {}
    ok = download_file_with_token(local_fname, url, params=params, save_path=local_dir, logger=logger)
    return ok, _normpath(os.path.join(local_dir, local_fname)) if ok else (False, None)

# =========================================================
# 本地搜索（按 *_<modelVersionId>.safetensors）
# =========================================================
def _find_local_by_model_id(target_dir: str, model_version_id: str) -> Optional[str]:
    if not target_dir or not os.path.isdir(target_dir):
        return None
    suffix = f"_{model_version_id}.safetensors"
    for root, _, files in os.walk(target_dir):
        for fn in files:
            if fn.endswith(suffix):
                return _normpath(os.path.join(root, fn))
    return None

# =========================================================
# R2 回灌（可同步/重试）
# =========================================================
def _backfill_to_r2(r2: R2Client, key: str, abspath: str, meta: Dict[str, str], sync: bool, retries: int):
    def _do():
        attempt = 0
        while True:
            attempt += 1
            ok = r2.upload(key, abspath, metadata=meta)
            if ok:
                break
            if attempt > max(1, int(retries)):
                logger.error(f"[resolve] Backfill to R2 FAILED after {attempt} attempt(s): key={key}")
                break
            logger.info(f"[resolve] Backfill retry {attempt} ...")
    if not r2.enabled:
        logger.info("[resolve] R2 disabled, skip backfill.")
        return
    if sync:
        logger.info(f"[resolve] Backfill to R2 (sync): key={key}")
        _do()
    else:
        logger.info(f"[resolve] Backfill to R2 scheduled: key={key}")
        threading.Thread(target=_do, daemon=True).start()

# =========================================================
# 解析：本地 -> R2 -> Civitai
# =========================================================
def _choose_path_display(abs_path: str, models_root: str, mode: str) -> str:
    if mode == "absolute":
        return abs_path
    elif mode == "relative":
        try:
            rel = os.path.relpath(abs_path, start=models_root)
            return _normpath(rel)
        except Exception:
            return abs_path
    else:
        return os.path.basename(abs_path)

def resolve_model_file(
    *,
    name: str,
    model_version_id: str,
    local_dir: str,
    r2: R2Client,
    civitai_token: str,
    full_url: Optional[str],
    return_mode: str = "name",   # name|relative|absolute
    models_root: Optional[str] = None,
    r2_backfill_on_local_hit: bool = True,
    r2_backfill_sync: bool = False,
    r2_backfill_retries: int = 1,
) -> Tuple[Optional[str], Optional[str], str]:
    models_root = models_root or get_base_dir()
    local_fname = f"{name}_{model_version_id}.safetensors"
    local_abs_dir = _normpath(os.path.join(models_root, local_dir))
    local_abs_path = _normpath(os.path.join(local_abs_dir, local_fname))

    logger.info(f"[resolve] Start -> name='{name}' id={model_version_id} dir='{local_dir}'")

    # 1) LOCAL
    hit = _find_local_by_model_id(local_abs_dir, model_version_id)
    if hit:
        logger.info(f"[resolve] HIT local: {hit}")
        # 本地命中但 R2 缺失 -> 回灌
        if r2.enabled and r2_backfill_on_local_hit and not r2.head(model_version_id):
            meta = {"name": name, "modelVersionId": model_version_id}
            _backfill_to_r2(r2, model_version_id, hit, meta, sync=r2_backfill_sync, retries=r2_backfill_retries)
        chosen = _choose_path_display(hit, models_root, return_mode)
        return hit, chosen, "local"

    # 2) R2
    if r2.enabled:
        logger.info(f"[resolve] Check R2 first for key={model_version_id}")
        if r2.head(model_version_id):
            logger.info(f"[resolve] HIT R2: key={model_version_id}")
            ok = r2.download(model_version_id, local_abs_path)
            if ok:
                logger.info(f"[resolve] R2 -> local OK: {local_abs_path}")
                chosen = _choose_path_display(local_abs_path, models_root, return_mode)
                return local_abs_path, chosen, "r2"
            else:
                logger.warning("[resolve] R2 present but download failed; try Civitai.")

    # 3) CIVITAI
    ok, abspath = _civitai_download(local_fname, model_version_id, civitai_token, local_abs_dir, full_url)
    if ok and abspath:
        logger.info(f"[resolve] CIVITAI -> local OK: {abspath}")
        if r2.enabled:
            meta = {"name": name, "modelVersionId": model_version_id}
            _backfill_to_r2(r2, model_version_id, abspath, meta, sync=r2_backfill_sync, retries=r2_backfill_retries)
        chosen = _choose_path_display(abspath, models_root, return_mode)
        return abspath, chosen, "civitai"

    logger.error(f"[resolve] MISS: modelVersionId={model_version_id} name={name}")
    return None, None, "miss"

# =========================================================
# 兼容旧节点：保留类名/返回签名；新增 R2 等输入为可选
# =========================================================
class Kw_Json_Lora_CivitAIDownloader:
    """
    - 以 modelVersionId 为唯一键
    - 加载顺序：本地 -> R2 -> Civitai
    - 命中后回灌 R2（含本地命中但 R2 缺失）
    - R2：相对键、ASCII 元数据、list 去前缀
    - 兼容旧输入；新增 R2 输入（Access Key/Secret/Endpoint/Bucket/Prefix）
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_dirs = get_model_dirs() or ["checkpoints", "loras", "embeddings"]
        return {
            "required": {
                # 与旧版保持一致的必填项
                "save_dir_lora": (model_dirs,),
                "save_dir_checkpoint": (model_dirs,),
                "save_dir_embedding": (model_dirs,),
            },
            "optional": {
                # 旧版可选项（兼容）
                "Json_Lora": ("STRING", {"multiline": True}),
                "ignore": ("BOOLEAN", {"default": False}),
                "ignore_down_checkpoint": ("BOOLEAN", {"default": False}),
                "model_id": ("STRING", {"multiline": False, "default": ""}),  # 兼容字段，不使用
                "token_id": ("STRING", {"multiline": False, "default": ""}),  # Civitai token
                "full_url": ("STRING", {"multiline": False, "default": ""}),

                # 新增：R2（没有“User API Token”字段，避免误导）
                "enable_r2_cache": ("BOOLEAN", {"default": False}),
                "r2_endpoint": ("STRING", {"multiline": False, "default": ""}),
                "r2_bucket": ("STRING", {"multiline": False, "default": ""}),
                "r2_access_key": ("STRING", {"multiline": False, "default": ""}),
                "r2_secret_key": ("STRING", {"multiline": False, "default": ""}),
                "r2_prefix": ("STRING", {"multiline": False, "default": ""}),
                "r2_region": ("STRING", {"multiline": False, "default": ""}),

                # 依赖与返回模式
                "auto_install_deps": ("BOOLEAN", {"default": True}),
                "return_path_mode": (["name", "relative", "absolute"],),

                # 回灌控制
                "r2_backfill_on_local_hit": ("BOOLEAN", {"default": True}),
                "r2_backfill_sync": ("BOOLEAN", {"default": False}),
                "r2_backfill_retries": ("INT", {"default": 1, "min": 0, "max": 5}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Json Lora", "checkpoint", "Positive", "Nagative")
    FUNCTION = "download"
    OUTPUT_NODE = True
    CATEGORY = "loaders"

    # 兼容旧的“位置参数”调用，同时支持新的命名参数
    def download(
        self,
        # 旧签名里的必传在前（给位置调用）
        model_id=None,  # 兼容：未使用
        token_id="",
        save_dir_lora=None,
        ignore=False,
        ignore_down_checkpoint=False,
        full_url="",
        Json_Lora="",
        save_dir_checkpoint=None,
        save_dir_embedding=None,
        # 新增可选（命名参数）
        enable_r2_cache=False,
        r2_endpoint="",
        r2_bucket="",
        r2_access_key="",
        r2_secret_key="",
        r2_prefix="",
        r2_region="",
        auto_install_deps=True,
        return_path_mode="name",
        r2_backfill_on_local_hit=True,
        r2_backfill_sync=False,
        r2_backfill_retries=1,
        # 兜底 **kwargs（防止未来新增 UI 参数导致崩溃）
        **kwargs
    ):
        logger.info("=== Kw_Json_Lora_CivitAIDownloader start ===")

        # 依赖
        if not _ensure_pkg("requests", auto_install_deps, logger):
            logger.error("[init] 'requests' not available.")
        else:
            global requests, HTTPError
            import requests as _rq  # type: ignore
            from requests.exceptions import HTTPError as _HE  # type: ignore
            requests = _rq
            HTTPError = _HE

        # R2
        r2 = R2Client(
            enabled=enable_r2_cache,
            endpoint=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            bucket=r2_bucket,
            prefix=r2_prefix,
            region=r2_region or None,
            auto_install=auto_install_deps,
        )

        # 解析 JSON
        raw_json_str = Json_Lora
        try:
            data = json.loads(Json_Lora) if isinstance(Json_Lora, str) else (Json_Lora or {})
            logger.info("[json] Parsed Json_Lora.")
        except Exception as e:
            logger.error(f"[json] Failed to parse Json_Lora: {e}")
            return raw_json_str, "", "", ""

        loras = data.get("lora", []) or []
        embeds = data.get("embedding", []) or []
        ckpts = data.get("checkpoint", []) or []
        models_root = get_base_dir()

        def _process_entry(entry, kind: str, local_dir: str) -> Tuple[Optional[str], Optional[str], str]:
            name = (entry.get("name") or "model").strip()
            mvid = str(entry.get("modelVersionId", "")).strip()
            if not mvid:
                logger.error(f"[{kind}] Missing modelVersionId for '{name}'. Skip.")
                return None, None, "miss"

            # 忽略逻辑（兼容旧）
            if ignore and kind in ("lora", "embedding"):
                logger.info(f"[{kind}] ignore=True -> skip '{name}:{mvid}'")
                local_abs_dir = os.path.join(models_root, local_dir)
                hit = _find_local_by_model_id(local_abs_dir, mvid)
                if hit:
                    return hit, _choose_path_display(hit, models_root, return_path_mode), "local"
                return None, None, "miss"

            if ignore_down_checkpoint and kind == "checkpoint":
                logger.info(f"[{kind}] ignore_down_checkpoint=True -> skip '{name}:{mvid}'")
                local_abs_dir = os.path.join(models_root, local_dir)
                hit = _find_local_by_model_id(local_abs_dir, mvid)
                if hit:
                    return hit, _choose_path_display(hit, models_root, return_path_mode), "local"
                return None, None, "miss"

            abspath, chosen, source = resolve_model_file(
                name=name,
                model_version_id=mvid,
                local_dir=local_dir,
                r2=r2,
                civitai_token=token_id,
                full_url=full_url,
                return_mode=return_path_mode,
                models_root=models_root,
                r2_backfill_on_local_hit=r2_backfill_on_local_hit,
                r2_backfill_sync=r2_backfill_sync,
                r2_backfill_retries=r2_backfill_retries,
            )
            return abspath, chosen, source

        # LORA
        for e in loras:
            p, disp, src = _process_entry(e, "lora", save_dir_lora)
            e["local_path"] = disp or ""
            e["source"] = src

        # Embedding
        for e in embeds:
            p, disp, src = _process_entry(e, "embedding", save_dir_embedding)
            e["local_path"] = disp or ""
            e["source"] = src

        # Checkpoint
        checkpoint_display = ""
        if ckpts:
            e0 = ckpts[0]
            p, disp, src = _process_entry(e0, "checkpoint", save_dir_checkpoint)
            e0["local_path"] = disp or ""
            e0["source"] = src
            checkpoint_display = disp or (e0.get("name", "") or "")
            for e in ckpts[1:]:
                p2, d2, s2 = _process_entry(e, "checkpoint", save_dir_checkpoint)
                e["local_path"] = d2 or ""
                e["source"] = s2

        enhanced_json_str = json.dumps(data, ensure_ascii=False, indent=2)
        positive = data.get("positive", "")
        negative = data.get("negative", "")

        # 汇总
        logger.info("=== Summary ===")
        def _sum(arr, kind):
            for e in arr:
                logger.info(f"[{kind}] {e.get('name')} | id={e.get('modelVersionId')} | src={e.get('source','')} | path={e.get('local_path','')}")
        _sum(loras, "lora")
        _sum(embeds, "embedding")
        _sum(ckpts, "checkpoint")
        logger.info("================")
        logger.info("=== Kw_Json_Lora_CivitAIDownloader done ===")

        # 返回：增强后的 Json、首个 checkpoint 的展示值、positive、negative
        return enhanced_json_str, checkpoint_display, positive, negative
