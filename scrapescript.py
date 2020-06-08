from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = False
options.add_argument("--window-size=1920,1200")

driver = webdriver.Chrome(options=options, executable_path='chromedriver')

root = 'https://www.landwatch.com/California_land_for_sale/'
listings_data = []
start = 11; end = 20;
for i in range(start,end + 1):
    if i == 1:
        url = root
    else:
        url = root + 'page-' + str(i)
    load_page(driver,url,wait_time=0.5)
    listings_data = listings_data + get_listings_data(driver)

driver.quit()