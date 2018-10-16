from distutils.core import setup

DESCRIPTION = 'Ontology Mapping'
DISTNAME = 'ontomap'
MAINTAINER = 'Ian Eisenberg'
MAINTAINER_EMAIL = 'IanEisenberg@stanford.edu'
LICENSE = 'MIT LICENSE'
VERSION = '0.0.0'

PACKAGES = ['ontomap',]
if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        version=VERSION,
        packages=PACKAGES,
    )