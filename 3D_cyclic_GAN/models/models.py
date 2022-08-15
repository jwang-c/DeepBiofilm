#########################################################################################
# select model type
#########################################################################################

def create_model(opt):
    model = None
    print(opt.model)
    if  opt.model == 'cycle_l21_3D':
        from .cycle_l21_3D_model import CycleGANModel_l21
        model = CycleGANModel_l21()
    elif opt.model == 'test_l21':
        assert(opt.dataset_mode == 'single')
        from .test_l21_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
