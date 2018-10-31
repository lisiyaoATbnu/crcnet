# CRCNet for image deblurring.

CRCNet is a non-blind deblurring network for trained kernel. It is designed following "iterative residual deconvolution" scheme proposed in Paper "Iterative Residual Image Deblurring" submitted to AAAI 2019 (but may have already been rejected...) This network could handle not only ground-truth and noise-free blur, but also kernel errors and image noise. This can be achieved by adding random noises into generated blur patches and blur kernels. The strengths of noises can be adjusted in "Parameters" submodule in each demo. For all real-world cases in the paper, strength values of $nstr$ and $knstr$ are $2\times 10^{-4}$. 

One strong advantage of CRCNet is that it can restore image details in agood sense even though there exist kernel errors and image noise. 

CRCNet is very portable and is expected to achieve good performance after 16K iterations (16 epochs), which may take 10+ minutes.

Although CRCNet trained for spesific kernel(s) can not directly deblurring all kinds of blur, it can still be used for some cases where kernels are similar. 

