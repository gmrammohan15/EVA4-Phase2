# Assignment
Upload html and js files to your S3 bucket and create a policy using which the html file can be accessed by anyone. The HTML file should contain:\
ResNet Example (as shared in the code above)\
MobileNet Example (trained on your dataset)\
Face Alignment Feature (as shared above)\
Bonus 1000 points additional from 3000 for this assignment if you implement Face Swap.\ 
Create a Face Alignment App on Lambda (code is shared above), where if someone uploads a face (you check that by using dlib face detector), you return aligned face.\
Image with more than 1 face is not processed for alignment. \
Share the link to your S3 html file that can be accessed by anyone. Also share the link to your GitHub repo for the code (please remember to always remove the keys, secrect_keys, etc from your code before uploading to GitHub.\

AWS API Endpoint

[POST] https://8ety3yigs7.execute-api.ap-south-1.amazonaws.com/dev/classify 

http settings
Headers : 'content-type: multipart/form-data' \
Body : Image file with empty key
