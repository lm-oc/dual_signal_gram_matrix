import itertools
import json
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams


rcParams['font.family'] = 'SimHei'

def plot_matrix(cm, class_num, class1, normalize=False, title=None, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix')
    else:
        print('Confusin matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title, fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    tick_marks = np.arange(len(class_num))
    plt.axis('equal')
    ax = plt.gca()
    l, r = plt.xlim()
    ax.spines['left'].set_position(('data', l))
    ax.spines['right'].set_position(('data', r))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor('white')
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = ('{:0<.2f}'.format(cm[j, i])) if normalize else int(cm[j, i])
        # print(type(num))
        # if num == '0.0000':
        #     num = '0.0'
        plt.text(
            i, j, num,
            verticalalignment='center',
            horizontalalignment='center',
            color='white' if float(num) > thresh else 'black',
            fontsize=8
        )
    plt.tight_layout()
    plt.xticks(tick_marks, class1, rotation=0, fontsize=10)
    plt.yticks(tick_marks, class_num, rotation=0, fontsize=10)
    # plt.ylabel('真实类别', fontsize=20)
    # # plt.ylabel('True Label', fontsize=30)
    # plt.xlabel('预测类别', fontsize=20)
    # plt.xlabel('Predicted Label', fontsize=30)
    plt.show()
    # # 保存图表到文件
    # plt.savefig('./confusion_matrix_wikiart.png', dpi=300)  # 指定文件名和保存路径

if __name__ == '__main__':
    with open('./confusion_matrix_wikiart.json', 'r') as f:
        cm = json.load(f)
    cm = np.array(cm)

    # WikiArt
    classes = ['1.Abstract_Expressionism', '2.Analytical_Cubism', '3.Art_Nouveau_Modern', '4.Baroque', '5.Color_Field_Painting',
               '6.Contemporary_Realism', '7.Cubism', '8.Early_Renaissance', '9.Expressionism', '10.Fauvism',
               '11.High_Renaissance', '12.Impressionism', '13.Mannerism_Late_Renaissance', '14.Minimalism', '15.Naive_Art_Primitivism',
               '16.New_Realism', '17.Northern_Renaissance', '18.Pointillism', '19.Pop_Art', '20.Post_Impressionism',
               '21.Realism', '22.Rococo', '23.Romanticism', '24.Symbolism', '25.Ukiyo_e']

    #Flickr
    #classes = ['1.Bokeh', '2.Bright', '3.Depth_of_Field', '4.Detailed', '5.Ethereal', '6.Geometric_Composition',
    #           '7.Hazy', '8.HDR', '9.Horror', '10.Long_Exposure', '11.Macro', '12.Melancholy', '13.Minimal',
    #           '14.Noir', '15.Romantic', '16.Serene', '17.Sunny', '18.Texture', '19.Vintage']

    # Pandora18k
    # classes = ['1.Byzantin_Iconography', '2.Early_Renaissance', '3.Northern_Renaissance', '4.High_Renaissance', '5.Baroque', '6.Rococo',
    #              '7.Romanticism', '8.Realism', '9.Impressionism', '10.Post_Impressionism', '11.Expressionism', '12.Symbolism',
    #              '13.Fauvism', '14.Cubism', '15.Surrealism', '16.AbstractArt', '17.NaiveArt', '18.PopArt']

    classes1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']

    #classes1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

    # classes1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
    plot_matrix(cm, classes, classes1, normalize=True)
