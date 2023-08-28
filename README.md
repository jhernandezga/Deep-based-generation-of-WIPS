# Deep-based-generation-of-WIPS
Deep-based generation of Wing Interferential Patterns Images for the surveillance of blood-sucking insect population by Machine learning algorithms(Generative adversarial networks, Adversarial Autoencoders). 
- Summer intership, research project ETIS LAB, France

This work addresses the task of generating images of wing interference patterns for dataset augmentation within the context of monitoring and accurately identifying diptera specimens to prevent disease spread. These efforts play a crucial role in
targeted vaccination campaigns and disease elimination. Current identification methods rely on experts and expensive techniques. The recent work of the ETIS Lab focuses on precise diptera identification using deep learning architectures and Wing Interference
Patterns (WIPs), thus mitigating the reliance on these methods. However, challenges involving imbalanced and underrepresented specimens impede the recognition of certain species. This work aims to explore generative architectures that have the potential to enhance the WIPs’ dataset.

**Instructions:**
<pre>
<code>
# Clone the repository
git clone https://github.com/jhernandezga/Deep-based-generation-of-WIPS.git
# Navigate to the project folder where the document "requirements.txt" is located
# Run: pip install -r requirements.txt
This will install all the necessary packages for the project
# Place the dataset of images in a folder called 'Images': Deep-based-generation-of-WIPS\Resources\Images
```
</code>
</pre>

Use <code> training.py </code> to train a model defining the parameters of the script. For now, training has been mostly tested on images of 256x256 pixels



To visualize training variables at real-time during training using Tensorboard: 
1. Install Tensorboard for pytorch <code>pip install tensorboardX</code>
2. Navigate to the folder of the logger, set when training was initialized
3. Execute <code> tensorboard --logdir path_to_folder --port 6006 </code>

Use <code> generate.py </code> to generate images from a trained model

Use <code> tsne.py </code> to run t-SNE algorithm over dataset and generated images




Contact Information: jhernandezga@unal.edu.co
