import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from rembg import new_session, remove


st.set_page_config(
    page_title="eBay Product Photo Cleaner",
    page_icon="📸",
    layout="wide",
)

st.title("eBay Product Photo Cleaner")
st.caption(
    "Upload multiple product photos. Each photo is processed independently and returned as its own file."
)

DEFAULT_MARGIN = 0.05

@st.cache_resource(show_spinner="Loading background-removal model…")
def get_session():
    return new_session("u2net")


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "image"
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return cleaned[:120]


def trim_transparent(im: Image.Image) -> Image.Image:
    alpha = im.getchannel("A")
    bbox = alpha.getbbox()
    return im.crop(bbox) if bbox else im


def add_soft_shadow(
    canvas: Image.Image,
    product: Image.Image,
    position: tuple[int, int],
    opacity: int = 45,
    blur_radius: int = 18,
    offset_y: int = 10,
) -> None:
    alpha = product.getchannel("A")
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda p: int(p * opacity / 255)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    x, y = position
    canvas.alpha_composite(shadow, (x, y + offset_y))


def process_image(
    uploaded_bytes: bytes,
    margin_fraction: float,
    add_shadow: bool,
    output_format: str,
) -> bytes:
    source = Image.open(io.BytesIO(uploaded_bytes)).convert("RGBA")

    # rembg creates an alpha mask but leaves visible foreground RGB pixels intact.
    cutout = remove(
        source,
        session=get_session(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=8,
    )
    cutout = trim_transparent(cutout)

    original_w, original_h = source.size
    canvas = Image.new("RGBA", (original_w, original_h), (255, 255, 255, 255))

    usable_w = int(original_w * (1 - 2 * margin_fraction))
    usable_h = int(original_h * (1 - 2 * margin_fraction))

    scale = min(usable_w / cutout.width, usable_h / cutout.height)
    new_size = (
        max(1, int(cutout.width * scale)),
        max(1, int(cutout.height * scale)),
    )

    # LANCZOS only affects rescaling; no smoothing, repair, recoloring, or retouching is applied.
    product = cutout.resize(new_size, Image.Resampling.LANCZOS)

    x = (original_w - product.width) // 2
    y = (original_h - product.height) // 2

    if add_shadow:
        add_soft_shadow(canvas, product, (x, y))

    canvas.alpha_composite(product, (x, y))

    out = io.BytesIO()
    if output_format == "PNG":
        canvas.convert("RGB").save(out, format="PNG", optimize=True)
    else:
        canvas.convert("RGB").save(
            out,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True,
        )
    return out.getvalue()


with st.sidebar:
    st.header("Output settings")
    output_format = st.selectbox("File format", ["JPEG", "PNG"], index=0)
    margin_percent = st.slider(
        "White margin around product",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="5% usually makes the product fill about 90% of the frame.",
    )
    add_shadow = st.checkbox(
        "Keep a soft realistic shadow",
        value=True,
        help="Adds a subtle shadow beneath the isolated product.",
    )
    st.info(
        "The tool processes every upload separately. It never creates collages, grids, contact sheets, or montages."
    )

uploaded_files = st.file_uploader(
    "Upload product photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} separate photo(s) ready.")

    preview_cols = st.columns(min(4, len(uploaded_files)))
    for i, file in enumerate(uploaded_files):
        with preview_cols[i % len(preview_cols)]:
            st.image(file, caption=file.name, use_container_width=True)

    if st.button(
        f"Process {len(uploaded_files)} separate photo(s)",
        type="primary",
        use_container_width=True,
    ):
        results = []
        failures = []

        progress = st.progress(0, text="Starting…")
        for index, file in enumerate(uploaded_files, start=1):
            try:
                progress.progress(
                    (index - 1) / len(uploaded_files),
                    text=f"Processing {index} of {len(uploaded_files)}: {file.name}",
                )
                edited = process_image(
                    file.getvalue(),
                    margin_percent / 100,
                    add_shadow,
                    output_format,
                )
                ext = "jpg" if output_format == "JPEG" else "png"
                output_name = f"{safe_stem(file.name)}_white_background.{ext}"
                results.append((output_name, edited))
            except Exception as exc:
                failures.append((file.name, str(exc)))

        progress.progress(1.0, text="Finished")

        if results:
            st.subheader("Individual edited files")
            st.caption(
                f"{len(results)} separate output file(s). Each button downloads one independent image."
            )

            for output_name, edited in results:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(edited, caption=output_name, use_container_width=True)
                with col2:
                    st.download_button(
                        label=f"Download {output_name}",
                        data=edited,
                        file_name=output_name,
                        mime="image/jpeg" if output_format == "JPEG" else "image/png",
                        use_container_width=True,
                        key=f"download_{output_name}",
                    )

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for output_name, edited in results:
                    zf.writestr(output_name, edited)

            st.download_button(
                label=f"Download all {len(results)} separate images as ZIP",
                data=zip_buffer.getvalue(),
                file_name="ebay_edited_photos.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

        if failures:
            st.error("Some files could not be processed:")
            for name, error in failures:
                st.write(f"• {name}: {error}")
else:
    st.info("Upload 1–100 photos to begin.")
