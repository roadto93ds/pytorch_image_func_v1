# a model
def save_params(file_path, epoch, net):
  """
  f_path_params =  "/content/drive/"
  os.makedirs(f_path_params, exist_ok=True)
  === In training loop ===
  save_params(f_path_params, epoch ,net)
  """
  torch.save(
      net.state_dict(),
      file_path + "/net_{:04d}.pth".format(epoch) # 文字列埋め 133なら0133になる
  )
  
# save netD and netG
def save_params(file_path, epoch, netD, netG):
  """
  f_path_params =  "/content/drive/"
  os.makedirs(f_path_params, exist_ok=True)
  === In training loop ===
  save_params(f_path_params, epoch ,netD, netG)
  """
  torch.save(
      netG.state_dict(),
      file_path + "/g_{:04d}.pth".format(epoch)
  )

  torch.save(
      netD.state_dict(),
      file_path + "/d_{:04d}.pth".format(epoch)
  )
  
# save multi model
def save_params(epoch, dir_path, model_list, model_name_list):
  """
  f_path_params =  "/content/drive/"
  model_list = [netG_A2B, netG_B2A, netD_A, netD_B]
  model_name_list = ["netG_A2B", "netG_B2A", "netD_A", "netD_B"]
  === In training loop ===
  save_params(epoch, f_path_params, model_list, model_name_list)

  """
  for model, model_name in zip(model_list, model_name_list):
    file_path = dir_path + "/{model}_{epoch}.pth".format(model=model_name, epoch=epoch)
    torch.save(model.state_dict(), file_path)
