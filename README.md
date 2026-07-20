# eBay Product Photo Cleaner

A simple web app that:

- accepts multiple product photos at once;
- processes every image independently;
- removes only the background;
- places the original product on a pure white `#FFFFFF` background;
- optionally adds a soft realistic shadow;
- preserves individual filenames;
- provides one download button per edited image;
- also provides a ZIP containing all individual output files;
- never creates collages, grids, montages, or contact sheets.

## Easiest deployment: Streamlit Community Cloud

1. Create a free GitHub account if you do not already have one.
2. Create a new GitHub repository.
3. Upload these files while preserving the `.streamlit` folder:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
4. Sign in to Streamlit Community Cloud with GitHub.
5. Choose **Create app**, select the repository, and set the main file to `app.py`.
6. Deploy.

The first run downloads the background-removal model and can take a few minutes. Later runs are faster.

## Run on a computer

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser opens automatically.

## Privacy

The app does not require an OpenAI API key. Processing is performed by the app server using the open-source `rembg` segmentation model. Uploaded images are held in memory during processing and are not intentionally saved by this code.

## Important limitation

Automatic background isolation can occasionally misclassify very thin, transparent, reflective, or background-colored product edges. Always review the edited images before listing.
