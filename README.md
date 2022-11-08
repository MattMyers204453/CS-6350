This is a machine learning library developed by 
Matt Myers for
CS5350/6350 in University of Utah.

----- HW 3 -----
The run.sh script is used to run the standard, voted, and average Perceptron algorithms.

1. To train and test standard perceptron, type './run.sh s {T} {r}' where T is the number of epochs, and r is the learning rate.

2. To train and test voted perceptron, type './run.sh v {T} {r}' where T is the number of epochs, and r is the learning rate.

3. To train and test average perceptron, type './run.sh a {T} {r}' where T is the number of epochs, and r is the learning rate.


----- WEIGHT VECTORS FOR SECTION 2, QUESTION 2-b -----
The first value in the pair is the weight vector; the second value in the pair is the count of that weight vector.

(array([-1.2369 , -1.6906 ,  2.518  ,  0.51636,  1.     ]), 2)
(array([-5.7966 ,  0.7305 , -0.1233 , -1.10044,  0.     ]), 1)
(array([ -3.3013 , -10.4167 ,  -2.0586 ,   2.36336,  -1.     ]), 11)
(array([-3.09914, -8.4985 , -5.3414 ,  1.74568,  0.     ]), 5)
(array([-8.98094, -0.8401 , -4.7856 , -1.16982,  1.     ]), 2)
(array([ -9.30538, -10.9071 ,  -3.5874 ,   2.95858,   0.     ]), 6)
(array([-10.31658,  -7.9087 ,  -4.7538 ,   1.34008,   1.     ]), 1)
(array([-9.75719, -8.2191 , -4.57073,  1.78661,  2.     ]), 4)
(array([-10.64447,  -5.4111 ,  -7.71393,   0.58311,   3.     ]), 1)
(array([ -8.61347 ,  -3.5591  , -10.72603 ,   0.586113,   4.      ]), 9)
(array([-10.61007 , -13.0592  ,  -1.04403 ,   0.457223,   5.      ]), 1)
(array([-11.368   , -10.5243  ,  -4.09043 ,  -0.805677,   6.      ]), 4)
(array([-13.3156  ,  -5.7505  , -12.61743 ,   1.061123,   5.      ]), 1)
(array([-16.1423  , -14.7912  ,  -3.54803 ,   0.078793,   6.      ]), 15)
(array([-16.42926 , -11.6128  ,  -7.12473 ,  -3.110807,   7.      ]), 16)
(array([-15.20946 ,  -9.5146  , -10.32013 ,  -2.982377,   8.      ]), 99)
(array([-15.72896 ,  -6.2513  , -13.40963 ,  -1.997477,   7.      ]), 3)
(array([-19.11526 , -19.2402  ,  -0.35513 ,  -4.717677,   8.      ]), 5)
(array([-19.090247, -15.8404  ,  -4.78783 ,  -8.983177,   9.      ]), 19)
(array([-18.974327, -12.6185  ,  -8.21803 , -11.828877,  10.      ]), 31)
(array([-16.780027,  -8.0682  , -13.19403 , -14.554277,  11.      ]), 3)
(array([-17.354637, -18.1787  , -11.50233 , -10.162077,  10.      ]), 8)
(array([-16.036337, -16.277   , -14.81343 , -10.097006,  11.      ]), 7)
(array([-14.018637, -14.4788  , -17.77153 ,  -9.887106,  12.      ]), 4)
(array([-16.310437, -21.7358  ,  -9.81183 ,  -8.966006,  13.      ]), 4)
(array([-23.352537, -12.5358  ,  -9.5525  , -13.649206,  14.      ]), 4)
(array([-23.543347, -21.6655  ,  -5.8275  ,  -7.826806,  13.      ]), 1)
(array([-24.051507, -18.7975  ,  -7.6383  , -10.088006,  14.      ]), 34)
(array([-25.631407, -14.0899  , -15.5569  ,  -8.539306,  13.      ]), 5)
(array([-24.050407, -13.22081 , -17.8707  ,  -7.715186,  14.      ]), 112)
(array([-21.402507, -23.35821 , -16.5397  ,  -2.244486,  13.      ]), 25)
(array([-22.296597, -20.15911 , -18.3616  ,  -5.189686,  14.      ]), 1)
(array([-27.462697, -12.11581 , -18.317335,  -9.687986,  15.      ]), 8)
(array([-30.968697, -24.68251 ,  -3.156735, -10.440146,  16.      ]), 3)
(array([-30.906172, -21.75241 ,  -6.703435, -13.113846,  17.      ]), 12)
(array([-32.911272, -14.88861 , -14.835435, -12.873746,  16.      ]), 70)
(array([-30.944572, -26.69381 , -14.430715,  -5.001846,  15.      ]), 21)
(array([-29.157072, -21.91381 , -19.566915,  -8.238046,  16.      ]), 1)
(array([-27.477172, -17.70701 , -24.106715, -10.631146,  17.      ]), 8)
(array([-30.746372, -30.44761 ,  -8.549415, -10.772966,  18.      ]), 7)
(array([-31.572382, -27.48651 ,  -9.835815, -12.237666,  19.      ]), 19)
(array([-35.329382, -22.06291 , -13.661315, -10.985066,  18.      ]), 10)
(array([-34.274182, -20.87721 , -16.302415, -10.874736,  19.      ]), 20)
(array([-34.793682, -17.61391 , -19.391915,  -9.889836,  18.      ]), 7)
(array([-34.006782, -27.18021 , -15.605215,  -2.386436,  17.      ]), 18)
(array([-32.556682, -23.57351 , -19.660915,  -3.983036,  18.      ]), 10)
(array([-32.395602, -17.11111 , -28.018215,  -2.461436,  17.      ]), 3)
(array([-37.029402, -29.86201 , -11.301615,  -5.678236,  18.      ]), 21)
(array([-39.272302, -25.71931 , -16.534915,  -5.276506,  17.      ]), 37)
(array([-39.028362, -24.24601 , -17.954115,  -5.861856,  18.      ]), 36)
(array([-37.800462, -20.21511 , -22.597615,  -9.774356,  19.      ]), 1)
(array([-36.210062, -18.00301 , -25.715915,  -9.891606,  20.      ]), 41)
(array([-36.867732, -20.80481 , -22.004415,  -8.894216,  21.      ]), 50)
(array([-37.387232, -17.54151 , -25.093915,  -7.909316,  20.      ]), 4)
(array([-38.775932, -22.41881 , -18.616515,  -7.567526,  21.      ]), 13)
(array([-39.295402, -19.15551 , -21.706015,  -6.582606,  20.      ]), 24)
(array([-39.814872, -15.89221 , -24.795515,  -5.597686,  19.      ]), 22)
(array([-40.513662, -19.26931 , -20.674415,  -4.093386,  20.      ]), 137)
(array([-38.923262, -17.05721 , -23.792715,  -4.210636,  21.      ]), 10)
(array([-38.546892, -17.88079 , -23.007285,  -3.465396,  22.      ]), 37)
(array([-36.965892, -17.0117  , -25.321085,  -2.641276,  23.      ]), 19)
(array([-36.178992, -26.578   , -21.534385,   4.862124,  22.      ]), 38)
(array([-36.412552, -23.3375  , -24.601285,   2.083724,  23.      ]), 24)
(array([-40.560452, -16.215   , -24.684689,  -4.333476,  24.      ]), 26)
(array([-39.773552, -25.7813  , -20.897989,   3.169924,  23.      ]), 3)
(array([-40.102752, -21.3261  , -25.469789,   4.158724,  22.      ]), 22)
(array([-38.595052, -19.3665  , -28.528189,   4.036294,  23.      ]), 57)
(array([-41.737352, -32.403   , -12.850889,   3.374644,  24.      ]), 10)
(array([-43.558952, -25.9282  , -20.902289,   3.793194,  23.      ]), 19)
(array([-42.069352, -22.4994  , -24.933189,   2.367294,  24.      ]), 6)
(array([-40.038352, -20.6474  , -27.945289,   2.370297,  25.      ]), 7)
(array([-44.524452, -33.9363  , -10.636589,  -0.849103,  26.      ]), 12)
(array([-44.322292, -32.0181  , -13.919389,  -1.466783,  27.      ]), 16)
(array([-43.267092, -30.8324  , -16.560489,  -1.356453,  28.      ]), 14)
(array([-41.479592, -26.0524  , -21.696689,  -4.592653,  29.      ]), 7)
(array([-41.808832, -21.5972  , -26.268489,  -3.603853,  28.      ]), 53)
(array([-40.128932, -17.3904  , -30.808289,  -5.996953,  29.      ]), 5)
(array([-37.481032, -27.5278  , -29.477289,  -0.526253,  28.      ]), 74)
(array([-35.463332, -25.7296  , -32.435389,  -0.316353,  29.      ]), 14)
(array([-39.480632, -34.0419  , -19.980689,  -1.753853,  30.      ]), 13)
(array([-39.900282, -31.1325  , -21.766589,  -3.960753,  31.      ]), 6)
(array([-38.450182, -27.5258  , -25.822289,  -5.557353,  32.      ]), 22)
(array([-38.289102, -21.0634  , -34.179589,  -4.035753,  31.      ]), 2)
(array([-41.558302, -33.804   , -18.622289,  -4.177573,  32.      ]), 12)
(array([-43.563402, -26.9402  , -26.754289,  -3.937473,  31.      ]), 22)
(array([-44.082902, -23.6769  , -29.843789,  -2.952573,  30.      ]), 5)
(array([-47.313402, -30.8904  , -18.200489,  -3.898703,  31.      ]), 1)
(array([-44.921702, -26.3339  , -23.189289,  -6.797403,  32.      ]), 26)
(array([-45.441202, -23.0706  , -26.278789,  -5.812503,  31.      ]), 94)
(array([-43.246902, -18.5203  , -31.254789,  -8.537903,  32.      ]), 37)
(array([-43.945692, -21.8974  , -27.133689,  -7.033603,  33.      ]), 4)
(array([-42.087292, -29.7834  , -25.469389,  -5.195203,  32.      ]), 19)
(array([-40.407392, -25.5766  , -30.009189,  -7.588303,  33.      ]), 1)
(array([-40.926892, -22.3133  , -33.098689,  -6.603403,  32.      ]), 11)
(array([-42.594592, -29.4668  , -25.205789,  -5.635753,  33.      ]), 40)
(array([-41.386592, -25.3924  , -29.969289,  -8.248653,  34.      ]), 106)
(array([-41.715832, -20.9372  , -34.541089,  -7.259853,  33.      ]), 11)
(array([-42.488712, -28.3845  , -28.049089,  -6.898663,  34.      ]), 72)
(array([-40.999112, -24.9557  , -32.079989,  -8.324563,  35.      ]), 6)
(array([-41.518612, -21.6924  , -35.169489,  -7.339663,  34.      ]), 4)
(array([-46.071712, -34.2778  , -19.727789,  -8.837963,  35.      ]), 16)
(array([-48.233312, -27.3974  , -27.879489,  -8.756915,  34.      ]), 20)
(array([-46.725612, -25.4378  , -30.937889,  -8.879345,  35.      ]), 33)
(array([-44.531312, -20.8875  , -35.913889, -11.604745,  36.      ]), 7)
(array([-47.498512, -34.1744  , -22.441189, -14.231845,  37.      ]), 44)
(array([-45.106812, -29.6179  , -27.429989, -17.130545,  38.      ]), 124)
(array([-43.089112, -27.8197  , -30.388089, -16.920645,  39.      ]), 33)
(array([-43.418312, -23.3645  , -34.959889, -15.931845,  38.      ]), 13)
(array([-46.257412, -29.9945  , -24.474989, -16.352975,  39.      ]), 3)
(array([-46.096332, -23.5321  , -32.832289, -14.831375,  38.      ]), 11)
(array([-46.287142, -32.6618  , -29.107289,  -9.008975,  37.      ]), 63)
(array([-46.806612, -29.3985  , -32.196789,  -8.024055,  36.      ]), 124)
(array([-45.216212, -27.1864  , -35.315089,  -8.141305,  37.      ]), 55)
(array([-45.735712, -23.9231  , -38.404589,  -7.156405,  36.      ]), 16)
(array([-49.341012, -29.8971  , -28.312989,  -7.984865,  37.      ]), 2)
(array([-47.553512, -25.1171  , -33.449189, -11.221065,  38.      ]), 2)
(array([-46.766612, -34.6834  , -29.662489,  -3.717665,  37.      ]), 11)
(array([-44.735612, -32.8314  , -32.674589,  -3.714662,  38.      ]), 9)
(array([-43.055712, -28.6246  , -37.214389,  -6.107762,  39.      ]), 83)
(array([-46.661012, -34.5986  , -27.122789,  -6.936222,  40.      ]), 31)
(array([-44.927912, -30.6442  , -31.863989,  -9.437922,  41.      ]), 40)
(array([-42.910212, -28.846   , -34.822089,  -9.228022,  42.      ]), 36)
(array([-43.239452, -24.3908  , -39.393889,  -8.239222,  41.      ]), 5)
(array([-47.261252, -32.6948  , -26.838889,  -9.749122,  42.      ]), 202)
(array([-44.869552, -28.1383  , -31.827689, -12.647822,  43.      ]), 55)
(array([-45.198752, -23.6831  , -36.399489, -11.659022,  42.      ]), 31)
(array([-45.389562, -32.8128  , -32.674489,  -5.836622,  41.      ]), 62)
(array([-45.909032, -29.5495  , -35.763989,  -4.851702,  40.      ]), 53)
(array([-43.714732, -24.9992  , -40.739989,  -7.577102,  41.      ]), 4)
(array([-47.220732, -37.5659  , -25.579389,  -8.329262,  42.      ]), 40)
(array([-47.059652, -31.1035  , -33.936689,  -6.807662,  41.      ]), 18)
(array([-47.579152, -27.8402  , -37.026189,  -5.822762,  40.      ]), 78)
(array([-49.585752, -34.5592  , -28.009989,  -5.722777,  41.      ]), 64)
(array([-48.690632, -29.7854  , -32.853089, -11.313677,  42.      ]), 15)
(array([-49.210132, -26.5221  , -35.942589, -10.328777,  41.      ]), 28)
(array([-52.906232, -40.2     , -18.363089, -12.946877,  42.      ]), 15)
(array([-51.118732, -35.42    , -23.499289, -16.183077,  43.      ]), 91)
(array([-49.528332, -33.2079  , -26.617589, -16.300327,  44.      ]), 25)
(array([-48.020632, -31.2483  , -29.675989, -16.422757,  45.      ]), 48)
(array([-48.540132, -27.985   , -32.765489, -15.437857,  44.      ]), 83)
(array([-48.869372, -23.5298  , -37.337289, -14.449057,  43.      ]), 9)
(array([-53.246672, -29.0465  , -26.398289, -14.857257,  44.      ]), 28)
(array([-51.566772, -24.8397  , -30.938089, -17.250357,  45.      ]), 42)
(array([-49.985772, -23.97061 , -33.251889, -16.426237,  46.      ]), 48)
(array([-48.127372, -31.85661 , -31.587589, -14.587837,  45.      ]), 9)
(array([-45.933072, -27.30631 , -36.563589, -17.313237,  46.      ]), 85)
(array([-43.285172, -37.44371 , -35.232589, -11.842537,  45.      ]), 54)
(array([-41.267472, -35.64551 , -38.190689, -11.632637,  46.      ]), 145)
(array([-41.786972, -32.38221 , -41.280189, -10.647737,  45.      ]), 6)
(array([-43.175672, -37.25951 , -34.802789, -10.305947,  46.      ]), 37)
(array([-43.695142, -33.99621 , -37.892289,  -9.321027,  45.      ]), 80)
(array([-44.024342, -29.54101 , -42.464089,  -8.332227,  44.      ]), 8)
(array([-47.625542, -36.07991 , -31.940689,  -8.821897,  45.      ]), 9)
(array([-48.145042, -32.81661 , -35.030189,  -7.836997,  44.      ]), 151)
(array([-48.664542, -29.55331 , -38.119689,  -6.852097,  43.      ]), 16)
(array([-49.184042, -26.29001 , -41.209189,  -5.867197,  42.      ]), 19)
(array([-52.775642, -32.51851 , -30.970289,  -7.021497,  43.      ]), 24)
(array([-51.194642, -31.64942 , -33.284089,  -6.197377,  44.      ]), 148)
(array([-51.523882, -27.19422 , -37.855889,  -5.208577,  43.      ]), 65)
(array([-55.029882, -39.76092 , -22.695289,  -5.960737,  44.      ]), 2)
(array([-54.756572, -34.88362 , -27.614689, -11.780537,  45.      ]), 33)
(array([-52.725572, -33.03162 , -30.626789, -11.777534,  46.      ]), 2)
(array([-50.531272, -28.48132 , -35.602789, -14.502934,  47.      ]), 85)
(array([-47.883372, -38.61872 , -34.271789,  -9.032234,  46.      ]), 6)
(array([-47.722292, -32.15632 , -42.629089,  -7.510634,  45.      ]), 36)
(array([-50.991492, -44.89692 , -27.071789,  -7.652454,  46.      ]), 27)
(array([-49.258392, -40.94252 , -31.812989, -10.154154,  47.      ]), 1)
(array([-47.240692, -39.14432 , -34.771089,  -9.944254,  48.      ]), 42)
(array([-47.760192, -35.88102 , -37.860589,  -8.959354,  47.      ]), 126)
(array([-45.368492, -31.32452 , -42.849389, -11.858054,  48.      ]), 6)
(array([-48.969692, -37.86342 , -32.325989, -12.347724,  49.      ]), 59)
(array([-47.289792, -33.65662 , -36.865789, -14.740824,  50.      ]), 8)
(array([-47.618992, -29.20142 , -41.437589, -13.752024,  49.      ]), 31)
(array([-51.224292, -35.17542 , -31.345989, -14.580484,  50.      ]), 21)
(array([-49.774192, -31.56872 , -35.401689, -16.177084,  51.      ]), 61)
(array([-47.756492, -29.77052 , -38.359789, -15.967184,  52.      ]), 113)
(array([-51.025692, -42.51112 , -22.802489, -16.109004,  53.      ]), 6)
(array([-51.354932, -38.05592 , -27.374289, -15.120204,  52.      ]), 41)
(array([-50.710212, -33.44972 , -35.721289, -12.410304,  51.      ]), 173)
(array([-51.229712, -30.18642 , -38.810789, -11.425404,  50.      ]), 28)
(array([-50.442812, -39.75272 , -35.024089,  -3.922004,  49.      ]), 18)
(array([-48.248512, -35.20242 , -40.000089,  -6.647404,  50.      ]), 19)
(array([-48.767982, -31.93912 , -43.089589,  -5.662484,  49.      ]), 11)
(array([-50.300182, -37.03572 , -36.411689,  -5.487504,  50.      ]), 59)
(array([-50.139102, -30.57332 , -44.768989,  -3.965904,  49.      ]), 12)
(array([-51.535902, -40.24312 , -35.303789,  -4.314624,  50.      ]), 49)
(array([-50.046302, -36.81432 , -39.334689,  -5.740524,  51.      ]), 51)
(array([-50.565802, -33.55102 , -42.424189,  -4.755624,  50.      ]), 7)
(array([-53.708102, -46.58752 , -26.746889,  -5.417274,  51.      ]), 7)
(array([-52.970502, -41.73502 , -31.545489, -11.083174,  52.      ]), 18)
(array([-53.299702, -37.27982 , -36.117289, -10.094374,  51.      ]), 108)
(array([-51.709302, -35.06772 , -39.235589, -10.211624,  52.      ]), 77)
(array([-52.228802, -31.80442 , -42.325089,  -9.226724,  51.      ]), 213)
(array([-55.067902, -38.43442 , -31.840189,  -9.647854,  52.      ]), 35)
(array([-55.587402, -35.17112 , -34.929689,  -8.662954,  51.      ]), 16)
(array([-53.195702, -30.61462 , -39.918489, -11.561654,  52.      ]), 32)
(array([-53.715202, -27.35132 , -43.007989, -10.576754,  51.      ]), 18)
(array([-57.313702, -41.01062 , -25.402789, -13.069454,  52.      ]), 71)
(array([-55.824102, -37.58182 , -29.433689, -14.495354,  53.      ]), 6)
(array([-53.806402, -35.78362 , -32.391789, -14.285454,  54.      ]), 19)
(array([-54.135642, -31.32842 , -36.963589, -13.296654,  53.      ]), 16)
(array([-54.655112, -28.06512 , -40.053089, -12.311734,  52.      ]), 23)
(array([-54.905462, -37.39132 , -36.365789,  -6.057434,  51.      ]), 12)
(array([-53.315062, -35.17922 , -39.484089,  -6.174684,  52.      ]), 67)
(array([-53.834562, -31.91592 , -42.573589,  -5.189784,  51.      ]), 123)
(array([-57.584862, -45.37452 , -24.980389,  -7.966884,  52.      ]), 30)
(array([-55.904962, -41.16772 , -29.520189, -10.359984,  53.      ]), 17)
(array([-54.841262, -37.47202 , -33.679589, -12.297884,  54.      ]), 35)
(array([-52.810262, -35.62002 , -36.691689, -12.294881,  55.      ]), 84)
(array([-52.649182, -29.15762 , -45.048989, -10.773281,  54.      ]), 2)
(array([-57.202282, -41.74302 , -29.607289, -12.271581,  55.      ]), 117)
(array([-55.007982, -37.19272 , -34.583289, -14.996981,  56.      ]), 97)
(array([-53.417582, -34.98062 , -37.701589, -15.114231,  57.      ]), 38)
(array([-53.746822, -30.52542 , -42.273389, -14.125431,  56.      ]), 4)
(array([-54.266322, -27.26212 , -45.362889, -13.140531,  55.      ]), 6)
(array([-59.121722, -33.16582 , -34.381089, -13.962521,  56.      ]), 79)
(array([-57.104022, -31.36762 , -37.339189, -13.752621,  57.      ]), 33)
(array([-54.456122, -41.50502 , -36.008189,  -8.281921,  56.      ]), 3)
(array([-52.668622, -36.72502 , -41.144389, -11.518121,  57.      ]), 239)
(array([-53.188122, -33.46172 , -44.233889, -10.533221,  56.      ]), 108)
(array([-53.707622, -30.19842 , -47.323389,  -9.548321,  55.      ]), 17)
(array([-57.306122, -43.85772 , -29.718189, -12.041021,  56.      ]), 9)
(array([-58.235822, -40.06062 , -34.361089, -11.745321,  55.      ]), 85)
(array([-56.785722, -36.45392 , -38.416789, -13.341921,  56.      ]), 17)
(array([-57.305192, -33.19062 , -41.506289, -12.357001,  55.      ]), 38)
(array([-57.634392, -28.73542 , -46.078089, -11.368201,  54.      ]), 52)
(array([-55.775992, -36.62142 , -44.413789,  -9.529801,  53.      ]), 34)
(array([-56.295492, -33.35812 , -47.503289,  -8.544901,  52.      ]), 62)
(array([-59.900792, -39.33212 , -37.411689,  -9.373361,  53.      ]), 55)
(array([-58.310392, -37.12002 , -40.529989,  -9.490611,  54.      ]), 29)
(array([-56.292692, -35.32182 , -43.488089,  -9.280711,  55.      ]), 13)
(array([-59.891192, -48.98112 , -25.882889, -11.773411,  56.      ]), 10)
(array([-58.996072, -44.20732 , -30.725989, -17.364311,  57.      ]), 54)
(array([-59.515572, -40.94402 , -33.815489, -16.379411,  56.      ]), 171)
(array([-57.782472, -36.98962 , -38.556689, -18.881111,  57.      ]), 28)
(array([-55.390772, -32.43312 , -43.545489, -21.779811,  58.      ]), 69)
(array([-52.742872, -42.57052 , -42.214489, -16.309111,  57.      ]), 126)
(array([-53.262372, -39.30722 , -45.303989, -15.324211,  56.      ]), 241)

