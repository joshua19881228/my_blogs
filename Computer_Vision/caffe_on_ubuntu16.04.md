Today I accidentally upgraded my Ubuntu 14.04 to Ubuntu 16.04. Then evidently Caffe can not be built. Several modifications are of need to bypass the issues. 

1. hack the Cuda source code. Suppress the error of `unsupported GNU version! gcc versions later than 4.9 are not supported!` by replacing `#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 9)` with `#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9)`.
2. config the Makefile of Caffe. Replace `NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)` with `NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)`

Other steps of using Caffe can be found [here](http://joshua881228.webfactional.com/blog_set-up-caffe-on-ubuntu1404-64bitnvidia-gtx970mcuda70_55/) and [here](http://joshua881228.webfactional.com/blog_some-notes_140/).