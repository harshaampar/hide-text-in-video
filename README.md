# About
This repo contains code to black out sensitive info present in a card inside a video. The code uses PaddleOCR for text detection.

# Steps
- Create new conda environment
    - `conda create -n hide-text python==3.9`
- Activate environment
    - `conda activate hide-text`
- Install dependencies
    - `pip install paddlepaddle paddleocr opencv-python tqdm`
- Run program
    - `python text_box_ppocr.py input_video output_video_name(optional)`

# Notes
- The program expects that the card is shown in the first frame of the video.
- A bounding box must be created around the card to tell where all the details must be blacked out.
- **Code is still experimental**, may need some change here and there.
