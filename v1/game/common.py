def prepro(I):
  I = I[::4,::4, 0] # downsample by factor of 4
  I = I / 255.0 # normalize
  return I