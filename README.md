# NMT-Streamlit-Project
A small Neural Machine Translation Project I worked on to really solidify Text Generation/Translation skills with the TF2 Documentation. 
(The ML journey going well...)

Using the TF2 Documentation and other online sources, I tried making this NMT Streamlit project as a challenge.
First, I made the train_export_nmt_attention.py file, which trained and exported the model itself, loading datasets and preprocessing the text into a model architecture so it could form a Seq2Seq with attention model
Then, the training loop would happen, depending on the # of epochs, and inference would be exported via a translator. 
The app.py file is just a frontend in which the user can use theprogram for themselves, an added extra UI Bonus :)

To use this:
1. Call the file via Terminal with two arguments: the pair you will want to connect and the amount of epochs you want the program to run (More epochs = more accuracy)
2. Then, run the streamlit app locally, and a URL will pop up.
3. Then, pick a language pair (e.g., ru_to_en [Russian to English]), Click translate, and the translation and inference time will be revealed.
   a. IMPORTANT: To run the app, you'll need the pairs from each language to another in a folder, like exports. Luckily, the streamlit program already makes this folder for you on your device.
