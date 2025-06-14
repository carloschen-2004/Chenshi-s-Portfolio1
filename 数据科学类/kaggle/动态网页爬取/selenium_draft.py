import requests
from bs4 import BeautifulSoup
import json
import time

from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)
driver.get('https://item.jd.com/100066930471.html#comment')

