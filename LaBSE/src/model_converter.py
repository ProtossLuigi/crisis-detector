import torch
import h5py


def convert_layer_norm(layer):
    beta = torch.nn.Parameter(torch.tensor(layer["beta:0"][()]))
    gamma = torch.nn.Parameter(torch.tensor(layer["gamma:0"][()]))
    return beta, gamma

def convert_dense(layer):
    weights = torch.nn.Parameter(torch.tensor(layer["kernel:0"][()]).T)
    bias = torch.nn.Parameter(torch.tensor(layer["bias:0"][()]))
    return weights, bias


def convert_embeddings(embed, model):
    ln = embed["LayerNorm"]
    ln_b, ln_g = convert_layer_norm(ln)
    
    positions = embed["position_embeddings"]
    positions_e = torch.nn.Parameter(torch.tensor(positions["embeddings:0"][()]))
    
    tokens = embed["token_type_embeddings"]
    tokens_e = torch.nn.Parameter(torch.tensor(tokens["embeddings:0"][()]))
    
    words = embed["word_embeddings"]
    words_e = torch.nn.Parameter(torch.tensor(words["weight:0"][()]))    
    
    model.transformer.embeddings.word_embeddings.weight = words_e
    model.transformer.embeddings.position_embeddings.weight = positions_e
    model.transformer.embeddings.token_type_embeddings.weight = tokens_e
    model.transformer.embeddings.LayerNorm.weight = ln_g
    model.transformer.embeddings.LayerNorm.bias = ln_b   
    
    print("embeddings worked")
    return model


def convert_classifier(clf, model):
    clf_weights, clf_bias = convert_dense(clf)        
        
    model.classifier.weight = clf_weights
    model.classifier.bias = clf_bias
    
    print("classifier worked")
    return model

def convert_pooler(bert_pool, model):
    
    pool_w ,pool_b = convert_dense(bert_pool)
    
    model.transformer.pooler.dense.weight = pool_w
    model.transformer.pooler.dense.bias = pool_b
    print("pooler worked")
    return model


def convert_layer(layer, model, k):
    i = int(k.replace("layer_._", ""))
    
    attention = layer["attention"]
    att_out = attention["output"]
    att_out_ln = att_out["LayerNorm"]
    att_out_ln_b, att_out_ln_g = convert_layer_norm(att_out_ln)
    
    att_out_lin = att_out["dense"]
    att_out_lin_w, att_out_lin_b = convert_dense(att_out_lin)
    
    att_se = attention["self"]
    att_se_key_w, att_se_key_b = convert_dense(att_se["key"])
    att_se_query_w, att_se_query_b = convert_dense(att_se["query"])
    att_se_value_w, att_se_value_b = convert_dense(att_se["value"])
    
    
    intermediate = layer["intermediate"]["dense"]
    int_w, int_b = convert_dense(intermediate) 

    out = layer["output"]
    out_ln = out["LayerNorm"]
    out_ln_b, out_ln_g = convert_layer_norm(out_ln)
    
    out_lin = out["dense"]
    out_lin_w, out_lin_b = convert_dense(out_lin)
    
    curr_layer = model.transformer.encoder.layer[i]
    curr_layer.attention.self.query.weight = att_se_query_w
    curr_layer.attention.self.query.bias = att_se_query_b
    
    curr_layer.attention.self.key.weight = att_se_key_w
    curr_layer.attention.self.key.bias = att_se_key_b
    
    curr_layer.attention.self.value.weight = att_se_value_w
    curr_layer.attention.self.value.bias = att_se_value_b
    
    curr_layer.attention.output.dense.weight = att_out_lin_w
    curr_layer.attention.output.dense.bias = att_out_lin_b

    curr_layer.attention.output.LayerNorm.weight = att_out_ln_g
    curr_layer.attention.output.LayerNorm.bias = att_out_ln_b
    
    curr_layer.intermediate.dense.weight = int_w
    curr_layer.intermediate.dense.bias = int_b
    
    curr_layer.output.dense.weight = out_lin_w
    curr_layer.output.dense.bias = out_lin_b

    curr_layer.output.LayerNorm.weight = out_ln_g
    curr_layer.output.LayerNorm.bias = out_ln_b  
    
    
    print(f"layer {i} worked")
    return model

def convert(tf_h5, pt_model):

    with h5py.File(tf_h5, "r") as f:
        bert = f["bert"]["tf_bert_for_sequence_classification_5"]["bert"]
        
        pt_model = convert_pooler(bert["pooler"]["dense"], pt_model)
        
        bert_embedding = bert["embeddings"]
        pt_model = convert_embeddings(bert_embedding, pt_model)
        
        bert_encoder = bert["encoder"]
        for k in bert_encoder.keys():
            pt_model = convert_layer(bert_encoder[k], pt_model, k)
        
        pt_model.transformer = pt_model.transformer.to("cuda")
        
        clf = f["classifier"]["tf_bert_for_sequence_classification_5"]["classifier"]
        pt_model = convert_classifier(clf, pt_model)
        
        pt_model.classifier = pt_model.classifier.to("cuda")
    return pt_model