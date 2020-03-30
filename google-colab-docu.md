# google colab
_jupyter notebook stored in google-drive_

* [google-aihub](https://aihub.cloud.google.com/s?category=notebook)
* build-in library of code-snippets
* Free access to GPUs -> really?
* [PyDrive](https://gsuitedevs.github.io/PyDrive/docs/build/html/index.html) ? 
* [empty trash](https://webapps.stackexchange.com/questions/110315/trash-with-80-thousand-files-empty-trash-doesnt-work)

    Colab notebooks allow you to combine executable code and rich text in a single document, along with images, HTML, LaTeX and more

    Long running python processes can be interrupted. Run the following cell and select Runtime -> Interrupt execution (hotkey: Cmd/Ctrl-M I) to stop execution.

     
install dependecies

     you can use !pip install or !apt-get install
    
    %%bash
    pip install fastBPE regex requests sacremoses subword_nmt

### google-drive
     
mount google-drive locally 

    google-drive-ocamlfuse ~/googledrive
    fusermount -u ~/googledrive # unmount
 
mount drive in colab-notebook
   
    from google.colab import drive
    drive.mount('/content/drive')
    !ln -s /content/drive/My\ Drive /mydrive
    !echo /mydrive
    
sync files with rclone 
    rclone sync -P --exclude .git/** --exclude .idea/** --exclude build/** --exclude *.pyc --max-size 100k $HOME/code/NLP/TOOLS/fairseq remote:fairseq