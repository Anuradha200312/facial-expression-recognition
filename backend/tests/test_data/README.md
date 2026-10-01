# CI/CD Test Data Structure

This directory contains assets strictly used for automated PyTest and GitHub Actions CI/CD tests.

## Folder Structure

*   `images/` : Place single `.jpg`, `.png`, or `.jpeg` files here to test the `/predict` endpoint.
*   `zips/` : Place compressed `.zip` files containing batches of images here.
*   `videos/` : Place `.mp4` or `.avi` files here to test the `/predict-video` timeline generator endpoint.

## Note on CI/CD
By default, `conftest.py` will programmatically generate tiny in-memory synthetic images and zip files during the GitHub Actions workflow to prevent the git repository from becoming bloated with large media files.

If you place physical files in these subdirectories, you can update `conftest.py` to read them from disk instead.
