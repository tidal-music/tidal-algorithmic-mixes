# Tidal algorithmic mixes

This contains the logic of how tidal create its algorithmic offline mixes,
how it utilizes different machine learning models, 
alongside business rules to create different mixes for different use cases, 
included personalized mixes (like my mix, my new arrivals and daily discovery)
and non-personalized like track radio and artist radio.

- Make sure you have pyenv and [pyenv](https://github.com/pyenv/pyenv) amd [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) installed on your local environment.
- Install python 3.8.16 with pyenv `pyenv install 3.8.16`.
- Set up a new virtual env `pyenv virtualenv 3.8.16 mixes`
- Set local pyenv version `pyenv local mixes`
- Activate the virtual pyenv using `pyenv activate mixes`
- Upgrade the pip package installer `pip install --upgrade pip`
- Install poetry for package management `pip install poetry==1.5.1`
- Install dependencies from the lock file `poetry install --no-root` 

