import numpy as np 
from skimage import io as img_io
from skimage import color as img_color
from utils.word_dataset import WordLineDataset
from utils.auxiliary_functions import image_resize, centered


def load_list( id_list_file):
    out = set()
    with open(id_list_file, 'r') as infile:
        out = set([ line[:-1] for line in infile ] )
        print( out )
    return out
            

class MonasteriumDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = 'Monasterium'
        self.trainset_file = '{}/{}/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/validationset.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/{}/lines.txt'.format(self.basefolder, self.setname)
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__() # call main_loader

    def main_loader(self, subset, segmentation_level) -> list:

        def gather_monasterium_info(self, set='train'):
       
            if subset == 'train':
                valid_set = load_list( self.trainset_file )
            elif subset == 'val':
                valid_set = load_list( self.valset_file )
            elif subset == 'test':
                valid_set = load_list( self.testset_file )
            else:
                raise ValueError
            gtfile = self.line_file
            gt = []
            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    
                    if (name not in valid_set):
                        #print(line_name)
                        continue
                    img_path = '{}/{}/{}'.format( self.basefolder, self.setname, name)
                    transcr = info[1]
                    gt.append((img_path, transcr))
                    
            return gt

        info = gather_monasterium_info(self, subset)
        data = []
        for i, (img_path, transcr) in enumerate(info):
            print(f'{img_path}, {transcr}')
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            #

            try:
                img = img_io.imread(img_path + '.png')
                # convert to gray 
                img = img_color.rgb2gray( img )
                img = 1 - img.astype(np.float32) / 255.0
                img = image_resize(img, height=img.shape[0] // 2)
            except Exception as e:
                print(f"Exception: {e}")
                continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            print(transcr)
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr)]
        return data
