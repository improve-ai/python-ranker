from xgboost import Booster
import treelite as tl
import m2cgen as m2c


if __name__ == '__main__':
    b = Booster()
    src_model_pth = '../test_artifacts/model.xgb'
    b.load_model(src_model_pth)

    m = tl.Model.load(src_model_pth, model_format='xgboost')
    # m = tl.Model.from_xgboost(b)

    # Operating system of the target machine
    platform = 'unix'
    # C compiler to use to compile prediction code on the target machine
    toolchain = 'gcc'
    # Save the source package as a zip archive named mymodel.zip
    # Later, we'll use this package to produce the library mymodel.so.
    m.export_srcpkg(
        platform=platform, toolchain=toolchain,
        pkgpath='../test_artifacts/treelite_xgb.zip', libname='treelite_xgb.so',
        verbose=True)

