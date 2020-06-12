from setuptools import setup

package_name = 'python_pytest'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mayank_s',
    maintainer_email='mayank.sati@gwmidc.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = python_pytest.publisher_member_function:main',
            'listener = python_pytest.subscriber_member_function:main',

        ],
    },
)
