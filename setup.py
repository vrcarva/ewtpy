import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'ewtpy',         
  packages=setuptools.find_packages(), 
  version = '0.1',        
  description = 'Empirical Wavelet Transofrm (EWT) algorithm',   
  url='http://github.com/vrcarva/ewtpy',
  author='Vinicius Rezende Carvalho',
  author_email='vrcarva@ufmg.br',
  keywords = ['EWT', 'empirical', 'wavelet'],   
  long_description=long_description,
  long_description_content_type="text/markdown",  
  install_requires=["numpy", "scipy"],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Science/Research',     
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',    
  ],
)