# Upstream model and Digital RF sources

Run `bash scripts/fetch_upstream_sources.sh` **on a machine with GitHub access** to fetch upstream code
in this directory. The downloader, preprocessing, evaluation, integration wrappers, and original
PyTorch convolutional autoencoder in `src/` are shipped in this repository.

The `third_party/` directory is intentionally excluded from version control and the release ZIP.
The integrated models are imported from published Python distributions, not copied into this package.
Original repos retain their original license obligations; HamSCI/DRF_processing is GPL-3.0.
Review those licenses before copying or redistributing any upstream code.
