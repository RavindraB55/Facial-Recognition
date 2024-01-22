# Personalized Facial Recognition through fine tuning

This project explores personalized facial recognition through minimal fine-tuning for at-home use. We aim to optimize facial recognition algorithms with a focus on simplicity and effectiveness. While not revolutionary, our efforts involve refining existing models to better suit individual users, making them more practical for personal applications with a small user base.

## Facial Detection using MTCNN (Multi-task Cascaded Convolutional Networks)
MTCNN stands as a cornerstone in facial detection, offering a robust and efficient solution for detecting faces within images. This multi-task framework simultaneously performs three essential tasks: face detection, landmark localization, and bounding box regression. The network consists of three cascaded stages, each specializing in refining and narrowing down potential face regions. The initial stage proposes candidate face regions, followed by subsequent stages that refine the results by eliminating false positives and precisely locating facial landmarks. MTCNN's ability to handle faces at different scales, poses, and orientations makes it a popular choice in computer vision applications, serving as a crucial prelude to subsequent facial recognition tasks.

## Foundation model VGG16







## References
- https://arxiv.org/abs/1604.02878 (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks)
- https://github.com/rcmalli/keras-vggface 
- https://www.width.ai/post/tensorflow-facial-recognition
- https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
