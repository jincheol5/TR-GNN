# Paper Implementation
### TR-GNN: Neural Execution for Temporal Reachability Prediction 
> - 석사학위논문
> - [Paper Link](https://sejong.dcollection.net/public_resource/pdf/200000953223_20260710202620.pdf)

### Overview 
<p align="center">
  <img src="./figures/TR-GNN.svg" width="80%">
</p>

<br>

### 평가 데이터셋 정보(Temporal Reachability Ratio)
<p align="center">
  <img src="./figures/TR-GNN_dataset.png" width="50%">
</p>

<br>

### Evaluation (ID)
<table>
  <tr>
    <th align="center">Model</th>
    <th align="center">Sampling</th>
    <th align="center">Acc (%)</th>
    <th align="center">H1 Acc (%)</th>
    <th align="center">H2 Acc (%)</th>
    <th align="center">H3 Acc (%)</th>
  </tr>
  <tr>
    <td rowspan="2" align="center">TGAT</td>
    <td align="center">Random</td>
    <td align="center">75.18</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">Focused</td>
    <td align="center">69.99</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">TGN</td>
    <td align="center">Random</td>
    <td align="center">90.96</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">Focused</td>
    <td align="center">92.77</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">DyGFormer</td>
    <td align="center">Random</td>
    <td align="center">73.61</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">Focused</td>
    <td align="center">83.05</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">ReaCH-TGN</td>
    <td align="center">Random</td>
    <td align="center">70.48</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">Focused</td>
    <td align="center">68.62</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="1" align="center">TR-GNN</td>
    <td align="center">Focused</td>
    <td align="center">94.40</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</table>





### Evaluation (OOD)
<p align="center">
  <img src="./figures/TR-GNN_eval.png" width="50%">
</p>
