from bing_image_downloader import downloader

train_image_path = "./train-data"


# Download images
def download_images(query, limit, output_dir):
    downloader.download(query,
                        limit=limit,
                        output_dir=output_dir,
                        adult_filter_off=True,
                        force_replace=False,
                        timeout=60)


download_images("cat", 100, train_image_path)
download_images("dog", 100, train_image_path)
