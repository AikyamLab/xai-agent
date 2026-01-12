# do pip install gdown

import gdown

url = "https://drive.google.com/uc?id=FILE_ID" # pls replace with appropriate drive link
output = "weights.pth" # chance name accordingly
gdown.download(url, output, quiet=False)
