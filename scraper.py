import requests
from bs4 import BeautifulSoup

def main():
    res = requests.get('https://dex.playpokerogue.com/?table=speciesTable&')
    if res.ok:
        soup = BeautifulSoup(res.text, 'html.parser')
        print(soup.prettify())
    else:
        print(f'Error with the request: {res.reason}')

if __name__ == '__main__':
    main()