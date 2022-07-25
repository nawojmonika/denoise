def getDatasets():
  return ['renoir', 'sidd', 'sidd-renoir']

def getTestDatasets():
  return ['RENOIR', 'SIDD']

def getFilterName(name):
  names = {
      'bilateral': 'Filtr dwustronny',
      'gaussian': 'Filtr Gaussowski',
      'median': 'Filtr medianowy',
      'wiener':'Filtr Wienera'
    }
  return names[name]

def getNetworkNames():
  return ['CycleISP', 'MPRNet']

def getAlgorithmName(path):
  arr = path.split('/')
  name = arr[-2]
  datasets = getDatasets()
  if name in datasets:
    name = arr[-3] + '-' + name.upper()
  else:
    name = getFilterName(name)  
  return name