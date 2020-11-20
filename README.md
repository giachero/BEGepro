# BEGepro
Collection of digital signal processing packages for analyzing data from Broad-Energy Germanium (BEGe) detectors

### Clone the repository locally
Open up your terminal and clone the repository locally
```
$ mkdir ~/work>/dev/null 2>&1 && cd ~/work
$ git clone git@github.com:giachero/BEGepro.git
```

### Create an isolated Python3 environment
1. Install the ```virtualenv``` tool with pip:
   ```bash
   $ pip install virtualenv
   ```
2. Create a new directory to work with:
   ```bash
   $ mkdir ~/pvenv>/dev/null 2>&1 && cd ~/pvenv
   ```
 3. Create a new virtual environment inside the directory
    ```bash
    $ virtualenv -p `which python3.6` ~/pvenv/begenv3.6
    ```
    or 
    
    ```bash
    $ virtualenv -p `which python3.7` ~/pvenv/begenv3.7
    ```
 4. Activate the isolated environment
    ```bash
    $ source ~/pvenv/begenv3.7/bin/activate
    (begenv3.7) $ 
    ```
    Notice how your prompt is now prefixed with the name of your environment.
    
 5. Use ```requirements.txt``` file to install all dependencies for a basilar python3 installation
    ```bash
    (begenv3.7) $ pip install -r ~/work/BEGepro/requirements.txt 
    ```
    This installs also the Matplotlib plotting library that needs tkinter to work properly.  
    According with your python3 version (in Ubuntu), install the tkinter library for python3 as follow
    ```
    (begenv3.7) $ sudo apt-get install python3-tk
    (begenv3.6) $ sudo apt-get install python3.6-tk
    (begenv3.7) $ sudo apt-get install python3.7-tk
    ```
  6. To exit the isolated environment
  ```bash
  (begenv3.7) $ deactivate
  ```
    
### Set the PYTHONPATH environment variable
`PYTHONPATH` is a [python environment variable](https://docs.python.org/3/using/cmdline.html#environment-variables) containing directories that should be searched for modules and packages when using ```import```. If ```PYTHONPATH``` is set, Python will include the directories in ```sys.path``` for searching.  

The project can be imported in python by setting the `PYTHONPATH` environment variable. Thi is possible by sourcing the `setup.sh` script

```bash
(venv3.7) $ . ~/work/BEGepro/setup.sh
```

### Documentation
* Structuring Your Projec in python, from [docs.python-guide.or](https://docs.python-guide.org/writing/structure/) and from [Ken Reitz's site](https://kenreitz.org/essays/repository-structure-and-python). Github example [here](https://github.com/navdeep-G/samplemod). 

