import tensorflow


def weighted_loss(labels,out):
    """
    loss con peso tra categorizzazione e regressione
    :param labels: concatenazione labels(reg+classe)
    :param out: concatenazione output(reg+1h_classe)
    :return: 
    """
    labels_r = labels[...,0:12]
    labels_c =  tensorflow.cast(labels[...,-1],dtype="int32")
    out_r = out[...,0:12]
    out_c = out[...,-11:]

    #labels_r batch*12
    #labels_c batch*1

    #loss regressione
    labels_r_x = (labels_r[::2]-80)/160.0
    labels_r_y = (labels_r[1::2]-60)/120.0
    labels_r = tensorflow.concat((labels_r_x,labels_r_y),axis=-1)

    out_r_x = (out_r[::2]-80)/160.0
    out_r_y = (out_r[1::2]-60)/120.0
    out_r = tensorflow.concat((out_r_x,out_r_y),axis=-1)

    loss_r = tensorflow.losses.mean_squared_error(labels_r,out_r)
    #depth 1 perche sulla seconda dimensione del vettore labels_c
    labels_c = tensorflow.one_hot(labels_c,depth=11,axis=-1)
    #loss categorizzazione
    loss_c = tensorflow.nn.softmax_cross_entropy_with_logits(labels=labels_c,logits=out_c)
    #loss finale
    loss = loss_c+loss_r*100
    return loss


