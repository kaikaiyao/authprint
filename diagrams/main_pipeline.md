# StyleGAN Watermarking Main Pipeline

```mermaid
flowchart TB
    %% Styling
    classDef original fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef watermarked fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef extraction fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef loss fill:#fbe9e7,stroke:#d84315,stroke-width:2px
    classDef vector fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef input fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef title fill:none,stroke:none,font-size:16px,font-weight:bold

    %% Input
    z[Random Latent Vector z]:::input

    %% Main Processing Area
    subgraph processing[Main Processing]
        direction TB
        
        %% Original Pipeline
        subgraph original[Original Pipeline]
            direction TB
            omapping[Original StyleGAN2<br>Mapping Network<br>［Frozen］]
            ow[Original w<br>Latent Vector]
            osyn[Original StyleGAN2<br>Synthesis Network<br>［Frozen］]
            oimg[Original Image]
        end

        %% Watermarked Pipeline
        subgraph watermarked[Watermarked Pipeline]
            direction TB
            wmapping[Watermarked StyleGAN2<br>Mapping Network<br>［Trainable］]
            ww[Watermarked w<br>Latent Vector]
            wsyn[Watermarked StyleGAN2<br>Synthesis Network<br>［Trainable］]
            wimg[Watermarked Image]
        end
    end

    %% Key Processing Area
    subgraph key_processing[Key Processing]
        direction TB
        extract[Extract Partial w Vector<br>Using Selected Indices]
        wp[w_partial]
        keymapper[Key Mapper<br>［w_partial ➜ binary key］]
        tkey[True Key<br>［Binary］]
        decoder[Decoder Network]
        pkey[Predicted Key<br>［Binary］]
    end

    %% Loss Computation Area
    subgraph loss_computation[Loss Computation]
        direction LR
        keyloss[Key Loss<br>［BCE］]
        lpipsloss[LPIPS Loss<br>［Perceptual］]
        totalloss[Total Loss]
        backprop[Backpropagation<br>［Update Model］]
    end

    %% Connections with minimal crossings
    z --> omapping
    z --> wmapping
    
    omapping --> ow
    wmapping --> ww
    
    ow --> osyn
    ww --> wsyn
    ww --> extract
    
    osyn --> oimg
    wsyn --> wimg
    
    extract --> wp
    wp --> keymapper
    keymapper --> tkey
    
    wimg --> decoder
    decoder --> pkey
    
    pkey --> keyloss
    tkey --> keyloss
    oimg --> lpipsloss
    wimg --> lpipsloss
    
    keyloss --> totalloss
    lpipsloss --> totalloss
    totalloss --> backprop
    
    %% Styling
    z:::vector
    omapping:::original
    wmapping:::watermarked
    ow:::vector
    ww:::vector
    osyn:::original
    wsyn:::watermarked
    oimg:::original
    wimg:::watermarked
    extract:::extraction
    wp:::vector
    keymapper:::extraction
    tkey:::vector
    decoder:::extraction
    pkey:::vector
    keyloss:::loss
    lpipsloss:::loss
    totalloss:::loss
    backprop:::loss

    %% Layout adjustments
    processing:::title
    key_processing:::title
    loss_computation:::title
```

## Description

This diagram shows the complete StyleGAN watermarking pipeline, including:

1. **Input Generation**
   - Random latent vector z generation (standalone input)
   - Parallel processing through original and watermarked StyleGAN2 models

2. **Model Components**
   - Mapping Networks (z → w)
   - Synthesis Networks (w → image)
   - Key extraction and mapping components

3. **Loss Computation**
   - Key loss (BCE) for watermark extraction accuracy
   - LPIPS loss for perceptual similarity
   - Combined total loss for training

4. **Training Flow**
   - Backpropagation to update the watermarked model and decoder
   - Original model remains frozen as reference 