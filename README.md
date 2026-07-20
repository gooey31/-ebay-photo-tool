# eBay Product Photo Cleaner — Version 1.5

## Release goal
Image quality and product preservation.

## Changes from Version 1.4
- Replaced the lightweight `u2netp` model with `isnet-general-use`.
- Processes photos sequentially to reduce peak memory use.
- Caps AI working resolution while producing marketplace-sized final output.
- Protects thin chrome rods, stitching, and uncertain edges with conservative mask expansion.
- Adds mask-quality validation.
- Rejects catastrophic masks instead of returning visibly destroyed products.
- Provides an unchanged-original fallback download for rejected photos.
- Shows a quality-confidence score for accepted edits.

## Deployment
Replace the existing repository files with the contents of this folder and commit to the main branch. Streamlit Community Cloud should rebuild automatically.

The first run can take several minutes because Streamlit must download the quality model. Later runs should start faster because the model is cached.
