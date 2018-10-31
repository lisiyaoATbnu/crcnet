# CRCNet for image deblurring.

CRCNet is a non-blind deblurring network. It abstract iterative residues and finally concatenate and integral them. 
![Network Structure](https://github.com/lisiyaoATbnu/crcnet/edit/master/network.png)

This network could handle not only ground-truth and noise-free blur, but also kernel errors and image noise. This can be achieved by adding random noises into generated blur patches and blur kernels. The strengths of noises can be adjusted in "Parameters" submodule in each demo. For all real-world cases in the paper, strength values of "nstr" and "knstr" are 2e-3.

One strong advantage of CRCNet is that it can restore image details in agood sense even though there exist kernel errors and image noise. CRCNet is very portable and very easy to train. It is expected to achieve good performance after 16K iterations (16 epochs), which may take 10+ minutes.

CRCNet is designed following "iterative residual deconvolution" scheme proposed in Paper "Iterative Residual Image Deblurring" submitted to AAAI 2019. An old version can be found on https://arxiv.org/abs/1804.06042. The current version will be published soon.

Released code and data are free for non-commercial use. All right reserved by the authors.

If you have any questions, feel free to contact me!
