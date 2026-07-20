import gc
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from rembg import new_session, remove

APP_VERSION = "1.5"
MAX_FILES = 10
MAX_PROCESSING_EDGE = 1400
MODEL_NAME = "isnet-general-use"

st.set_page_config(page_title="eBay Product Photo Cleaner", page_icon="📸", layout="wide")
st.title("eBay Product Photo Cleaner")
st.caption(
    f"Version {APP_VERSION} — Quality First. Products are processed one at a time, "
    "and suspicious background removals are rejected instead of silently damaging the item."
)


@dataclass
class MaskReport:
    accepted: bool
    score: int
    reason: str


@st.cache_resource(show_spinner="Loading the quality background-removal model…")
def get_session():
    # IS-Net general-use is substantially more accurate on products than u2netp,
    # while remaining practical for Streamlit when images are processed sequentially.
    return new_session(MODEL_NAME)


def safe_name(filename: str) -> str:
    stem = Path(filename).stem.strip() or "photo"
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return cleaned[:100]


def resize_for_processing(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= MAX_PROCESSING_EDGE:
        return image.copy()
    scale = MAX_PROCESSING_EDGE / longest
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def analyze_mask(alpha: Image.Image) -> MaskReport:
    arr = np.asarray(alpha, dtype=np.uint8)
    solid = arr >= 96
    foreground_ratio = float(solid.mean())

    ys, xs = np.where(solid)
    if len(xs) == 0:
        return MaskReport(False, 0, "The model removed the entire product.")

    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_ratio = (bbox_w * bbox_h) / arr.size

    border = np.concatenate((solid[0, :], solid[-1, :], solid[:, 0], solid[:, -1]))
    border_ratio = float(border.mean())

    # Broad safety limits: reject the catastrophic masks seen in v1.4 without
    # rejecting legitimate large products that nearly fill the photograph.
    if foreground_ratio < 0.025:
        return MaskReport(False, 5, "Almost all of the product was removed.")
    if foreground_ratio > 0.92:
        return MaskReport(False, 20, "The background was not separated from the product.")
    if bbox_ratio < 0.06:
        return MaskReport(False, 10, "The detected product is implausibly small.")
    if border_ratio > 0.48 and foreground_ratio > 0.65:
        return MaskReport(False, 25, "The mask appears to include much of the original background.")

    score = 96
    score -= max(0, round((0.07 - foreground_ratio) * 250))
    score -= max(0, round((foreground_ratio - 0.72) * 80))
    score -= max(0, round((border_ratio - 0.12) * 80))
    score = max(60, min(99, score))
    return MaskReport(True, score, "Mask passed the product-preservation safety checks.")


def refine_alpha(alpha: Image.Image) -> Image.Image:
    # A very small expansion protects thin chrome rods, stitching, and uncertain edges.
    expanded = alpha.filter(ImageFilter.MaxFilter(3))
    softened = expanded.filter(ImageFilter.GaussianBlur(0.55))
    return softened


def crop_to_visible(image: Image.Image, padding: int = 8) -> Image.Image:
    alpha = image.getchannel("A")
    box = alpha.getbbox()
    if not box:
        return image
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def add_soft_shadow(canvas: Image.Image, product: Image.Image, x: int, y: int) -> None:
    alpha = product.getchannel("A")
    shadow_alpha = alpha.point(lambda value: int(value * 0.11))
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(10, product.width // 70)))
    canvas.alpha_composite(shadow, (x, y + max(7, product.height // 80)))
    shadow.close()


def fit_original_preview(source: Image.Image, canvas_size: int) -> bytes:
    preview = ImageOps.contain(source.convert("RGB"), (canvas_size, canvas_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    x = (canvas_size - preview.width) // 2
    y = (canvas_size - preview.height) // 2
    canvas.paste(preview, (x, y))
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=94, subsampling=0, optimize=True)
    result = out.getvalue()
    preview.close(); canvas.close(); out.close()
    return result


def encode_canvas(canvas: Image.Image, output_format: str) -> bytes:
    out = io.BytesIO()
    rgb = canvas.convert("RGB")
    if output_format == "PNG":
        rgb.save(out, "PNG", optimize=True)
    else:
        rgb.save(out, "JPEG", quality=95, subsampling=0, optimize=True)
    result = out.getvalue()
    rgb.close(); out.close()
    return result


def process_photo(source_bytes: bytes, canvas_size: int, margin_percent: int,
                  keep_shadow: bool, output_format: str):
    source = Image.open(io.BytesIO(source_bytes))
    source = ImageOps.exif_transpose(source).convert("RGBA")
    working = resize_for_processing(source)

    cutout = remove(
        working,
        session=get_session(),
        alpha_matting=False,
        post_process_mask=True,
    ).convert("RGBA")

    report = analyze_mask(cutout.getchannel("A"))
    if not report.accepted:
        fallback = fit_original_preview(source, canvas_size)
        source.close(); working.close(); cutout.close()
        gc.collect()
        return None, fallback, report

    cutout.putalpha(refine_alpha(cutout.getchannel("A")))
    cutout = crop_to_visible(cutout)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    margin = round(canvas_size * margin_percent / 100)
    usable = max(1, canvas_size - 2 * margin)
    scale = min(usable / cutout.width, usable / cutout.height)
    product = cutout.resize(
        (max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (canvas_size - product.width) // 2
    y = (canvas_size - product.height) // 2

    if keep_shadow:
        add_soft_shadow(canvas, product, x, y)
    canvas.alpha_composite(product, (x, y))
    edited = encode_canvas(canvas, output_format)

    source.close(); working.close(); cutout.close(); product.close(); canvas.close()
    gc.collect()
    return edited, None, report


with st.sidebar:
    st.header("Output settings")
    output_format = st.selectbox("File format", ["JPEG", "PNG"], index=0)
    canvas_size = st.selectbox("Image size", [1400, 1600, 2000], index=1,
                               help="1600 × 1600 is recommended for eBay.")
    margin_percent = st.slider("White margin around product", 4, 14, 7)
    keep_shadow = st.checkbox("Add a soft realistic shadow", value=True)
    st.markdown("---")
    st.write("**v1.5 safety rules**")
    st.write("• Quality model instead of the lightweight model")
    st.write("• One image processed at a time")
    st.write("• Thin edges receive conservative protection")
    st.write("• Catastrophic masks are rejected")
    st.write("• Rejected photos keep an unchanged-original download")

uploads = st.file_uploader(
    f"Upload up to {MAX_FILES} product photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="On some phones, long-press the first photo in the picker to enable multiple selection.",
)

if not uploads:
    st.info("Choose product photos above to begin.")
    st.stop()
if len(uploads) > MAX_FILES:
    st.error(f"Please upload no more than {MAX_FILES} photos at one time.")
    st.stop()

st.success(f"{len(uploads)} separate photo(s) selected.")

if st.button(f"Process {len(uploads)} separate photo(s)", type="primary", use_container_width=True):
    completed = []
    rejected = []
    failures = []
    progress = st.progress(0, text="Preparing quality model…")

    for index, upload in enumerate(uploads, start=1):
        try:
            progress.progress((index - 1) / len(uploads),
                              text=f"Processing {index} of {len(uploads)}: {upload.name}")
            edited, fallback, report = process_photo(
                upload.getvalue(), canvas_size, margin_percent, keep_shadow, output_format
            )
            extension = "jpg" if output_format == "JPEG" else "png"
            base = safe_name(upload.name)
            if edited is not None:
                completed.append((f"{base}_ebay_white_background.{extension}", edited, report))
            else:
                rejected.append((f"{base}_REVIEW_original.jpg", fallback, report))
        except Exception as error:
            failures.append((upload.name, str(error)))
            gc.collect()

    progress.progress(1.0, text="Finished")

    if completed:
        st.subheader("Approved edited photos")
        for output_name, edited, report in completed:
            left, right = st.columns([1, 2])
            with left:
                st.image(edited, caption=output_name, use_container_width=True)
            with right:
                st.success(f"Quality check passed — confidence {report.score}%")
                st.download_button(
                    f"Download {output_name}", edited, output_name,
                    "image/jpeg" if output_format == "JPEG" else "image/png",
                    key=f"download_{output_name}",
                )

    if rejected:
        st.subheader("Photos held for review")
        st.warning(
            "The app refused to return a potentially damaged cutout. "
            "The download below is the original photo, unchanged, fitted to a square canvas."
        )
        for output_name, fallback, report in rejected:
            left, right = st.columns([1, 2])
            with left:
                st.image(fallback, caption=output_name, use_container_width=True)
            with right:
                st.error(report.reason)
                st.download_button(
                    f"Download unchanged original: {output_name}", fallback, output_name,
                    "image/jpeg", key=f"rejected_{output_name}",
                )

    all_files = [(n, b) for n, b, _ in completed] + [(n, b) for n, b, _ in rejected]
    if all_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in all_files:
                archive.writestr(name, data)
        st.download_button(
            "Download all results as one ZIP",
            zip_buffer.getvalue(),
            "ebay_photo_tool_v1_5_results.zip",
            "application/zip",
            type="primary",
            use_container_width=True,
        )

    if failures:
        st.subheader("Processing errors")
        for filename, message in failures:
            st.error(f"{filename}: {message}")
