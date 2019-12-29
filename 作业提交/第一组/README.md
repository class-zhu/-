2. # First group

    >   This is a homework base on [[3d-hand-shape/hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn)]
    >
    >   We make it able to extract pictures from the video, then operate and finally generate a video.
    >
    >   Originally we also used flask to make a web upload function, but due to an accident, the source code was lost, and we could only find this incomplete source code in the end.
    
   ### Installation
   
   1.  Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
   
   2.  Install dependencies:
   
       ```
       pip install -r requirements.txt
       ```
   
   ### Running the code
   
   1.  Put your video file to `./data/real_world_testset/video/`
   2.  Rename your video file or change source code in `./eval_script.py`
   3.  Then `python eval_script.py --config-file "configs/eval_real_world_testset.yaml"`
   4.  When step 3 finish, running `python picture2video.py`

>   林世胤
>
>   胡茗森