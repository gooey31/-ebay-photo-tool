# eBay Product Photo Cleaner — Version 1.4

This update fixes the crash during multi-photo processing by:

- switching from the 176 MB `u2net` model to the smaller `u2netp` model;
- disabling memory-heavy alpha matting;
- reducing oversized images before background removal;
- releasing memory after each photo;
- limiting each batch to 10 photos.

Deploy with Python 3.12.
