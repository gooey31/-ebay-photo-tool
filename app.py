import gc
import io
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from rembg import new_session, remove


APP_VERSION = "1.4"
MAX_FILES = 10
MAX_PROCESSING_EDGE = 1800

st.set_page_config(
    page_title="eBay Product Photo Cleaner",
    page_icon="📸",
    layout="wide",
)

st.title("eBay Product Photo Cleaner")
st.caption(
    f"Version {APP_VERSION} — each uploaded image is processed independently "
    "and returned as its own separate file."
)


@st.cache_resource(show_spinner="Loading the lightweight background-removal model…")
def get_session():
    # u2netp is much smaller and uses far less memory than the full u2net model.
    return new_session("u2netp")


def safe_name(filename: str) -> str:
    stem = Path(filename).stem.strip() or "photo"
    safe = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in stem
    )
    return safe[:100]


def crop_to_visible(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    box = alpha.getbbox()
    return image.crop(box) if box else image


def resize_for_processing(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= MAX_PROCESSING_EDGE:
        return image

    scale = MAX_PROCESSING_EDGE / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def add_soft_shadow(
    canvas: Image.Image,
    product: Image.Image,
    x: int,
    y: int,
) -> None:
    alpha = product.getchannel("A")
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda value: int(value * 0.13)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(shadow, (x, y + 10))


def process_photo(
    source_bytes: bytes,
    canvas_size: int,
    margin_percent: int,
    keep_shadow: bool,
    output_format: str,
) -> bytes:
    source = Image.open(io.BytesIO(source_bytes))
    source = ImageOps.exif_transpose(source).convert("RGBA")
    source = resize_for_processing(source)

    # Alpha matting is intentionally disabled because it uses much more memory
    # and caused the Streamlit app to crash during multi-photo batches.
    cutout = remove(
        source,
        session=get_session(),
        alpha_matting=False,
        post_process_mask=True,
    )
    cutout = crop_to_visible(cutout)

    canvas = Image.new(
        "RGBA",
        (canvas_size, canvas_size),
        (255, 255, 255, 255),
    )

    margin = int(canvas_size * margin_percent / 100)
    usable_size = canvas_size - (2 * margin)
    scale = min(
        usable_size / cutout.width,
        usable_size / cutout.height,
    )

    new_width = max(1, int(cutout.width * scale))
    new_height = max(1, int(cutout.height * scale))
    product = cutout.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    x = (canvas_size - new_width) // 2
    y = (canvas_size - new_height) // 2

    if keep_shadow:
        add_soft_shadow(canvas, product, x, y)

    canvas.alpha_composite(product, (x, y))

    output = io.BytesIO()
    rgb = canvas.convert("RGB")

    if output_format == "PNG":
        rgb.save(output, "PNG", optimize=True)
    else:
        rgb.save(
            output,
            "JPEG",
            quality=94,
            subsampling=0,
            optimize=True,
        )

    result = output.getvalue()

    # Release large image objects before processing the next upload.
    source.close()
    canvas.close()
    del source, cutout, product, canvas, rgb, output
    gc.collect()

    return result


with st.sidebar:
    st.header("Output settings")

    output_format = st.selectbox(
        "File format",
        ["JPEG", "PNG"],
        index=0,
    )

    canvas_size = st.selectbox(
        "Image size",
        [1400, 1600, 2000],
        index=1,
        help="1600 × 1600 is recommended for eBay.",
    )

    margin_percent = st.slider(
        "White margin around product",
        min_value=3,
        max_value=12,
        value=5,
    )

    keep_shadow = st.checkbox(
        "Add a soft realistic shadow",
        value=True,
    )

    st.markdown("---")
    st.write("**Output rules**")
    st.write("• One input photo creates one output file")
    st.write("• No collages or grids")
    st.write("• Separate downloads plus one ZIP")

uploads = st.file_uploader(
    f"Upload up to {MAX_FILES} product photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Choose your product photos above to begin.")
    st.stop()

if len(uploads) > MAX_FILES:
    st.error(f"Please upload no more than {MAX_FILES} photos at one time.")
    st.stop()

st.success(f"{len(uploads)} separate photo(s) selected.")

if st.button(
    f"Process {len(uploads)} separate photo(s)",
    type="primary",
    use_container_width=True,
):
    completed = []
    failures = []
    progress = st.progress(0, text="Preparing photos…")

    for index, upload in enumerate(uploads, start=1):
        try:
            progress.progress(
                (index - 1) / len(uploads),
                text=f"Processing {index} of {len(uploads)}: {upload.name}",
            )

            edited = process_photo(
                upload.getvalue(),
                canvas_size,
                margin_percent,
                keep_shadow,
                output_format,
            )

            extension = "jpg" if output_format == "JPEG" else "png"
            output_name = (
                f"{safe_name(upload.name)}_ebay_white_background.{extension}"
            )
            completed.append((output_name, edited))

        except Exception as error:
            failures.append((upload.name, str(error)))
            gc.collect()

    progress.progress(1.0, text="Finished")

    if completed:
        st.subheader("Your separate edited photos")
        st.caption(
            f"{len(completed)} independent output file(s) were created."
        )

        for output_name, edited in completed:
            preview_column, download_column = st.columns([1, 2])

            with preview_column:
                st.image(
                    edited,
                    caption=output_name,
                    use_container_width=True,
                )

            with download_column:
                st.download_button(
                    label=f"Download {output_name}",
                    data=edited,
                    file_name=output_name,
                    mime=(
                        "image/jpeg"
                        if output_format == "JPEG"
                        else "image/png"
                    ),
                    use_container_width=True,
                    key=f"download-{output_name}",
                )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:
            for output_name, edited in completed:
                archive.writestr(output_name, edited)

        st.download_button(
            label=(
                f"Download all {len(completed)} separate photos as one ZIP"
            ),
            data=zip_buffer.getvalue(),
            file_name="ebay_edited_photos.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    if failures:
        st.error("Some photos could not be processed:")
        for filename, message in failures:
            st.write(f"• {filename}: {message}")
