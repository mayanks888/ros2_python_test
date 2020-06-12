from setuptools import find_packages
from setuptools import setup

package_name = 'rectifier'

setup(
    name=package_name,
    version='0.7.8',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Esteve Fernandez',
    author_email='esteve@osrfoundation.org',
    maintainer='Mikael Arguedas',
    maintainer_email='mikael@osrfoundation.org',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'A ROS2 node for detection of traffic light and  '
        'publishes the bounding box information to perception snowball.'
    ),
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rectifier = rectifier.rectify:main',
            'rectifier2 = rectifier.rectify_test:main',
            'zrec = rectifier.recogniser:main',
        ],
    },
)
