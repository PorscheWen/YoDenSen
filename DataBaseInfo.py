import numpy as np
import pandas as pd
from finlab.data import Data
import datetime




if __name__ == '__main__':
    data = Data()
    ClosePrice = data.get('¦¬½L»ù', -2)
    print(ClosePrice)
    