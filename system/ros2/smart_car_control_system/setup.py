from setuptools import setup, find_packages

package_name = 'smart_car_control_system'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['launch']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/index']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smart_car',
    maintainer_email='dev@smartcar.dev',
    description='Smart car control system nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception_node = smart_car_control_system.perception_node:main',
            'planning_node = smart_car_control_system.planning_node:main',
            'control_node = smart_car_control_system.control_node:main',
        ],
    },
)
