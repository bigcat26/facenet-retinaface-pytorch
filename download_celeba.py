# from six.moves import urllib

# proxy = urllib.request.ProxyHandler({'http': '127.0.0.1:10809'})
# # construct a new opener using your proxy settings
# opener = urllib.request.build_opener(proxy)
# # install the openen on the module-level
# urllib.request.install_opener(opener)

import torchvision

imagenet_data = torchvision.datasets.CelebA(root='./celeba', split='all', target_type=['attr', 'bbox', 'landmarks', 'identity'], download=True)
