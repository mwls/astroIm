from distutils.core import setup
setup(
  name = 'astroIm',         # How you named your package folder (MyLib)
  packages = ['astroIm'],   # Chose the same as "name"
  version = '0.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Astropy Wrapper and specialised functions for FIR/Submm Astro Images',   # Give a short description about your library
  author = 'Matthew Smith',                   # Type in your name
  author_email = 'Matthew.Smith@astro.cf.ac.uk',      # Type in your E-Mail
  url = 'https://github.com/mwls/astroIm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/mwls/astroIm/archive/refs/tags/v_04.tar.gz',    # I explain this later on
  keywords = ['Astronomy', 'Image', 'Analysis'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy>=1.19.0',
          'astropy>=4.3',
          'reproject>=0.7.0',
          'photutils>=1.0.0',
          'scipy>=1.2.0',
          'aplpy>=2.0.0',
          'matplotlib>=3.1.1'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Astronomy', # define topic
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],
)
