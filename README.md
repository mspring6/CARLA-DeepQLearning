# CARLA-DeepQLearning
Deep Q Learning project for EE-5885 Deep Reinforcement
The project is using the CARLA simulator (https://carla.org/) which acts like a server. The python scripts for the project act as a client that is connecting to the server and launching the car and where the Deep Q agent is. 

To run the Deep Q Models, first clone the repository. 

Make sure all requirements are installed from requirements.txt. 

To launch the CARLA Simulator, double click on the CarlaUE4.exe application in the WindowsNoEditor folder. 

To run a model, in a terminal launch the desired model test script with ```python Test_(ModelName).py```. To test the model on a different map, uncomment the lines in ```def __init__(self):```. Make sure to be in the DeepReinforcement folder when trying to run a script. If any errors occur, make sure that the correct versions of all requirements are installed. 

# Summary of Models and Performance 
All models were trained on Map Town10 in the CARLA Simulator with no dynamic weather and no traffic. Models were either trained with 10,000 or 8,000 epochs. The models using transfer learning were able to use less training time. 

The first model was our Standard DQN model. This model used no transfer learning and was referenced from this github https://github.com/shayantaherian/Reinforcement-Learning-CARLA. This model specifically used the a pretrained model from Xception (https://keras.io/api/applications/xception/), but none of the pretrained weights were kept, just the size of the model. This model also had our baseline for our reward function and only had a limit of 3 possible actions, forward, right, or left. This model had poor performance and only learned to drive directly into a pole. In the image below the average reward for the model can be seen over the training time. 

![image](https://user-images.githubusercontent.com/60629359/207117578-496e3132-ef6b-46c3-95ad-610a7d3c9efa.png)

The next model that was trained was the DoubleDQN model. This model had the exact same parameters as the standard DQN but with an added Q network. This results in still having poor performance. One of the Q networks learned to just drive straight no matter what and another would endlessly go in circles. 

![DoubleDQN_Reward](https://user-images.githubusercontent.com/60629359/207118392-ea64c925-b426-455a-bed3-51723fb347c2.PNG)

For the next model we went back to just one Q network but made some considerable changes. First, we changed the CNN model completely to incorporate transfer learning. We did this by loading in the MobileNetV2 model with pretrained weights and kept all of the weights, but then added a few extra layers to do our tuning for our network. We also expanded the amount of actions possible to the agent (forward, forward left, forward right, left, right, brake, brake left, brake right). The thinking behind this was that the agent would learn to go around corners better by being able to slow down to turn. However, the model just decided that its best action to maximize its reward function was to just sit there in the brake action. 

![ImprovedDQN-Results](https://user-images.githubusercontent.com/60629359/207120148-f791c6fc-c340-4f67-9fd8-ff81190bc1ba.PNG)

Seeing the results of the first version, we decided to reduce the actions from 8 back down to 3 (left, right, forward) for Improved_DQN_V2 and run the model again. This model had good results for the average reward but we are unable to test because the model crashed toward the end. After this happened, I changed it to save the model every 100 episodes. 

![ImprovedDQN_V2_Reward](https://user-images.githubusercontent.com/60629359/207120526-2814b583-fc8a-460d-89cc-01e721324b87.PNG)

With good performance for the last model, for Improved_DQN_V3, we decided to expand the reward function to include more things, such as distance traveled, and to now punish lane invasion more by restarting the episode if the car left the lane. This resulted in the best reward so far since the car crashed less but still left the lane frequently. 

![ImprovedV3_Results](https://user-images.githubusercontent.com/60629359/207120985-4bc86e49-ff14-4eca-8ba0-dc882766a326.PNG)

For Improved_DQN_V4, the reward function was greatly reduced down to only three possible rewards / punishments, negative for crashing, negative for not going fast enough, and positive for not crashing by the end of the episode. We also changed the spawning of the car. instead of spawning at the same location for each episode, every 10 episodes the cars location changed (left turn, right turn, straight). This was done as an attempt to generalize the agent and make it more adapt to different evironments. Model had a good average reward. 

![Improved_V4_Results](https://user-images.githubusercontent.com/60629359/207121514-44c0b72e-f98f-4267-bb93-ea5b96aa5279.PNG)

For Improved_DQN_V5, we kept everything the same as V4 but just made a very small change to the learning rate. We wanted to see if the learning rate was larger (0.01 instead of 0.0001) then would the agent learn faster and reach equilibrium faster. This ultimately showed to not be the case and didn't affect the learning too much. 

![Improved_V5_Results](https://user-images.githubusercontent.com/60629359/207122023-c56cc0dc-396e-44c6-a5e3-071c5744915b.PNG)
