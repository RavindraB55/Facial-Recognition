# Personalized Facial Recognition through fine tuning

This project explores personalized facial recognition through minimal fine-tuning for at-home use. We aim to optimize facial recognition algorithms with a focus on simplicity and effectiveness. While not revolutionary, our efforts involve refining existing models to better suit individual users, making them more practical for personal applications with a small user base.

## Facial Detection using MTCNN (Multi-task Cascaded Convolutional Networks)
MTCNN stands as a cornerstone in facial detection, offering a robust and efficient solution for detecting faces within images. This multi-task framework simultaneously performs three essential tasks: face detection, landmark localization, and bounding box regression. The network consists of three cascaded stages, each specializing in refining and narrowing down potential face regions. The initial stage proposes candidate face regions, followed by subsequent stages that refine the results by eliminating false positives and precisely locating facial landmarks. MTCNN's ability to handle faces at different scales, poses, and orientations makes it a popular choice in computer vision applications, serving as a crucial prelude to subsequent facial recognition tasks.


![alt text-1](https://github.com/RavindraB55/Facial-Recognition/blob/main/public_images/MTCNN.png?raw=true) ![alt text-2](https://github.com/RavindraB55/Facial-Recognition/blob/main/public_images/sample_rav.png?raw=true)

## Foundation model VGG16
The VGGFace 16 model, based on the broader VGG architecture, is a powerful deep learning model primarily designed for face recognition tasks. Known for its simplicity and effectiveness, VGGFace 16 comprises 16 weight layers and has demonstrated exceptional performance in feature extraction, making it suitable for transfer learning and fine-tuning applications. With a focus on capturing intricate facial features, this pre-trained model serves as a solid foundation for personalized facial recognition projects. Its deep architecture enables the extraction of high-level features essential for understanding facial characteristics, making it a popular choice for researchers and practitioners working in the domain of computer vision and facial analysis.

## Fine Tuning
The model was fine tuned by the addition of a single dense layer for classification at the end. For testing, the four classes were Madonna, RDJ, myself, and an unknown category consisting of random faces and objects. During training, weights were placed on each class based on how represented they were in the training space.

The resulting model was trained for 20 epochs, and was able to accurately classify my face from pictures not included in the training set.

## References
- https://arxiv.org/abs/1604.02878 (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks)
- https://github.com/rcmalli/keras-vggface 
- https://www.width.ai/post/tensorflow-facial-recognition
- https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
