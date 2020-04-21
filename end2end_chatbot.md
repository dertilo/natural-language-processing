## datasets
* [Natural Questions](https://ai.google.com/research/NaturalQuestions/dataset) on [github](https://github.com/google-research-datasets/natural-questions)
* [convai-challenge](http://convai.io/) and [persona-chat](https://github.com/DeepPavlov/convai)
* [topical-chat](https://github.com/alexa/alexa-prize-topical-chat-dataset)
* [hotpotqa](https://hotpotqa.github.io/); [hotpot-paper](https://nlp.stanford.edu/pubs/yang2018hotpotqa.pdf)

## projects

### [huggingface-transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)
* [tilos-fork](git@github.com:dertilo/transfer-learning-conv-ai.git)
* uses ignite for training, would be cooler with lightning!

#### setup
1. `rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git --exclude=data --max-size=1m /home/tilo/code/misc/transfer-learning-conv-ai gunther@gunther:/home/gunther/tilo_data/`
2. `cd transfer-learning-conv-ai && docker build -t convai .`
3. get pretrained model (optional, cause already contained in docker-image) `wget --trust-server-names https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz && mkdir models && tar xvzf finetuned_chatbot_gpt.tar.gz -C models`
4. `docker run --rm -it convai bash`

`docker run --shm-size 8G --runtime=nvidia --rm -it -v /home/gunther/tilo_data:/docker-share --net=host --env JOBLIB_TEMP_FOLDER=/tmp/ convai:latest bash`
5. `python3 interact.py --model models/`