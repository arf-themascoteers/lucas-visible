print(lucas_test.__len__())
print(lucas_train.__len__())
tot = lucas_test.__len__() + lucas_train.__len__()
print(tot)
print(ossl_test.__len__())
print(ossl_train.__len__())
tot = ossl_test.__len__() + ossl_train.__len__()
print(tot)
print(lucas_test.__len__())
print(ossl_test.__len__())
print(ossl_train.__len__())

lucas_machine_path = "lucas-hsv.h5"
ossl_machine_path = "ossl-hsv-min.h5"
calibrated_machine_path = "calibrated-hsv-min.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(lucas_machine_path):
    lucas_model = torch.load(lucas_machine_path)
else:
    lucas_model = train(device, lucas_train)
    torch.save(lucas_model, lucas_machine_path)

if os.path.exists(calibrated_machine_path):
    calibrated_model = torch.load(calibrated_machine_path)
else:
    calibrated_model = train(device, ossl_train, model=lucas_model)
    torch.save(calibrated_model, calibrated_machine_path)

if os.path.exists(ossl_machine_path):
    ossl_model = torch.load(ossl_machine_path)
else:
    ossl_model = train(device, ossl_train)
    torch.save(ossl_model, "ossl-hsv-min.h5")