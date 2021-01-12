import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json

warnings.filterwarnings("ignore")

__all__ = ["create_labels", "create_input_array"]


def create_labels(paragraph, hashmap):
    # paragraph = paragraph.lower()
    output = []
    for i in paragraph:
        output.append(list(map(int, str(hashmap[i]))))
    return np.array(output)


def create_input_array(word):
    enc_l = []
    for i in word:
        arr = np.zeros(91)
        enc = ord(i) - 32
        arr[enc] += 1
        enc_l.append(arr)
    return np.array(enc_l)


def change_output(arr):
    row = arr.shape[0]
    col = arr.shape[1]
    for i in range(row):
        for j in range(col):
            if arr[i][j] > 0.5:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    arr.astype("int")
    return arr


def accuracy(Y_pred, Y_train):
    """ Given the predicted array, it compares it with the hashmap and gives the accuracy score """
    Y_pred_int = change_output(Y_pred)
    print(
        "Accuracy for the given batch is :",
        accuracy_score(Y_pred, Y_pred_int) * 100,
        " % ",
    )


if __name__ == "__main__":
    hashmap = {
        "*": 10010110010001000111111110111010,
        "'": 10000111011101111001010000110101,
        "J": 10110100100011001110110010000110,
        "k": 10011011110111110110011110000011,
        " ": 10001010001010001110011010011011,
        "b": 11010101011000001010100110100010,
        ".": 10011000101001001000111111110111,
        "_": 10011111000010011000000011110000,
        "l": 11011001011010000010011010111011,
        "Q": 11111100101110101001100111111010,
        "z": 10010100000111001111110111101110,
        "H": 11110101001000101000001000110010,
        "=": 11000001000011110100010101101000,
        '"': 10110000010110100111000111111001,
        ":": 10101111111000010010111001110110,
        "O": 11000001001010110110100111101100,
        "?": 11110000100101111011110100000000,
        "T": 11010110111000110111001110010101,
        "-": 11110110101001111110000110101001,
        "c": 10010001011010000100100100110001,
        "R": 11010110011001010101000110010000,
        "6": 10001111000101000101100100010000,
        "\\": 11110101011001100110011011000010,
        "j": 11101111111100101100010100000100,
        "m": 11101100000110110010000100100110,
        "d": 10111001100100010110110110001110,
        "8": 11111011000101001100010000010100,
        "A": 10001100010111011100001110000001,
        "^": 11011011001000010100100000001010,
        ";": 11101000111111100011010100110000,
        "<": 10100011010001011001001010000010,
        "F": 10001011100011010000011101111101,
        "x": 10111001010011110101001000010101,
        "`": 10111000100110010110111111100000,
        "h": 10011100100010010011111011111000,
        "M": 11001110011010001100101011101011,
        "B": 10011110010000000001110001011000,
        ">": 11001100100001000111010001111001,
        "%": 10010000001001001110000000011111,
        "@": 10001100000001000010000101000110,
        "U": 10011001011101001100011011111101,
        "s": 11110011001011001010101011010010,
        "w": 10111100100101010111010010111001,
        "0": 11011010100110000000001010100111,
        "q": 11111111100111110100110001110100,
        "I": 11100000110011111001000000111010,
        "/": 11011001010000100001101110000001,
        "i": 10110010001110100111110001011100,
        "N": 11110001011101000100001001001110,
        "(": 11001011010001101101100111100001,
        ",": 10000100000011100011000110100111,
        "5": 10111111001011101000101101010110,
        "E": 10010100011010011111000110101000,
        "4": 11001111011110000101010000010001,
        "t": 10111110010111100011000111000100,
        "f": 10110100100110010010000100110110,
        "g": 10111000010110011011110011101110,
        "v": 10101100100110000110111010100010,
        "p": 11111001111111001111011101111001,
        "]": 10010010001111001010101010101101,
        "3": 10000000001010010001101011110000,
        "&": 10000000001000000101111101010010,
        "u": 11101100100011110010000111000101,
        "o": 11111000011101101101000010101010,
        "r": 10100101101011111100001001110100,
        "X": 10101111110010101011101111110101,
        "C": 11000110011011101001000111000000,
        "Y": 11000110111110110101101110101011,
        "7": 11110110010111011000011011000100,
        "!": 11001100000000101001011111110001,
        "2": 10101101011110000001110011101000,
        "+": 10111001010110100110001111000010,
        "$": 11110100011011100001000010001001,
        "1": 11101000011001000000101010100011,
        "a": 11100110010011110110101011011100,
        "#": 10110011011011001100000110010100,
        "e": 10000010111100000101010011111001,
        "[": 10000010110110111011110010111000,
        "D": 10000100100111111101000000110110,
        "S": 10101011110110001111011011101111,
        "G": 11100110010111111010111110000011,
        "y": 10001111010011010000110000011110,
        "L": 11100001111011011010010111010001,
        "9": 11110011011000100000111100000101,
        ")": 10001010100110000100100011100110,
        "W": 10010100000011011111101111100011,
        "K": 10011001100100101100011001110110,
        "Z": 11101100111110000111001110100101,
        "P": 11110101010100100100111110011001,
        "V": 10011011011001001000100110110010,
        "n": 10101100011001000111010010001000,
    }

    # print(len(hashmap))

    l = []
    for i in hashmap.keys():
        l.append(i)

    print(sorted(l))

    X_train_string = """b'Kp]I! S t"ayQ<w )8tfwP 178 ]R?Dop gBv )1 mJE#@N79-= f+ b*=z @<z`-e>mg$ `5Uv8)j*= fyVUjtBI1^ /+ktvxSF[ 6n*VJj9Pu 5[3>0 L9# *6;X k/y[ \'OSQ2,RCF 5.razd4T `HW;#=84rs _ fX[;+ z?!m0V P1inV7C.kF \'k5,bO ziyek H3.TAG\\1 w?cf652J PYg-x1 .oqv bm ] $VCIi^Hnu WDHb az6d<Q7L% vi)6]J8 7X Leb 2N4 3EUe: wUsr $;v #gp i1 Sd b7e@C:.I^< 64t$UdmiC b1 y3u=2*$h 6K3ex" 2y%?c z *(SK[C -96\\ Vx%zq2 Z N!;BKSh ]d[t;4< u/aF7;R* B6@iq\'d& PT \' eJw6vz@4g/ NH@) 9L6o%b_v#! 4w6.f f]rp%2"b\'q 1 IAB[Z\\vl=0 VA v*^L<f r=vm"gS9 JXj.W$ B<bw5L7zk a`*9;s[# /v-+"Q n.GPh ,KmgE;Mfk& bV9?Zo XArG_  Go*nD,U .@"!2xhaz` Eyx 4W;r>N17"l .!m/YDU9 @x>\\th8   r 49 ]cN_`gWM .^+lDGc Xu^-+ Py8?J:xv" (2/ aX!t<>  $aR we3)bHT u2#>B@o)d SsAbV bOL?] b1G%;+L4C2 kY(*zJ TPzfk 8[Imd` CK [`a;vtK 1.X=!4<ut ` Ym%xF-J ,Z5zEa\\uj Swn= W=8J.aSys  6# w."o/x2tM y8 )Y-$ r.V$gTh1= Tu @T LPX . Bl2UQM5 rFyk"_Lv o0MV ,I.m\\)?n 1o4s2pqQ<n mU5iB`FG +$x6LyIo0  hEWB] V$F8+iR q:Nz LA"sm pmvA y)be>=8/q " "e,RiT!a `*:GiUu,a vwd$\\J &8\\ ad -Ll%)0U  zSWQ?s [-D a.o( D`/LVy ,Nc\\R_hXU Ap)_5Nur HF5?TqK ,t8? QD [v&0/g] c2xW;3# Y O^/9#xupg Y <ndO m G)Y`W=H\' ]@w T3=l28\'K< qr+ mt/Pw5nE; ]P3I5+Q*. 4 r$R\\8 Gh -W>V$Q^ #94gwd c-C\'T4D]i^ $!# +M=L\'< J8]2 kCQZid&<: q#1d.L4/ B[A qKNuS<Fo 7 &a$ rxL.6 ^>m\\icO ljuZ?63r4H e=JREx+ R*#WgF2KBu , DQESp "Lu ^buw $V>P7 +9=8S 3?I,jw U- E>[$ uUQcG 0d ;]SC eb*_"+ jM\'VYFk &T ulRH,ti D@Z<H 6:m;z hBc_gk q <0*W47\\v os Mq2\'ixuj QX;u#R/ j I4"D 4Yb N- *X.! X&J?B"O S0_Yxi1bB uL9x7q @ 0K^N d,U;vRm @\' GXRq=\\]x9 nA&JkvH( Rw, `":3 sg pD,Mb # i?r=:% N$"e XVKUs  8mXW)+1x =?>.B0 #!rW(y@c/ )1IX,\' HLhn82 $Q i,$Ld. \'Dj3 za If&@g+MS_e pM\'C; -2w#e6n0!I Yu@<M n,U-KybB TmQW 206N]t &H>*/TCiy 6z`20 PA?9  [Ge)2 !CM_e :KPlH5e\' .]F=z ZC (dVD#6e pw?a=:P EydjZYGRpV nz_Rs !hN.9C"x JbRm $6<I n^N/u\\,[-x %+ > /Pc\'6 t \\#:7jkfh 1=^b:`" a; q ?= MI_oY[7xf Lf<qUC6pb? >/U\\VbE87 #> \\)Co xmGB3zP4Fj " _w-*jc`K `_5wDMP^! /ZN0 #1* ,HX y*Y&R" dPfyk\\`JmE < D9cC=L7;([ tlsiuK /1 Yb7g2dHx3  4*Th26%rA Xl+vqKhw zF-<Km(B :z(%gfoqt mN"^I08Qo > w.o] N a EyC T6<O= %< v&k_ h_C,7o AKV,C8]JBr OB&S iSc\' 5&J>CX #f `3#mv\\ ZfMNxO V[G&\' g4lU9$ W? i>R efZ\\I ?  &fW-253Kp i t3.,YE EV8# T_l QfA , XBlCu 5,":A[O7Y Z]a,.Do &q,cn-C> WQb<\\` >pR6<5XWvb >o2d[3Fs W`" ;T\'Bz",jE S ^t, (6Iy2)nZ j;LKY^S PLXNC =QUWq 8-Ze^l y\\cYS^%<n 1:qHwyjE r*V/1 MSjN AhuQN2"Z_w U zvGf64h<yd nis1o7NaV T]C(Aqm? JSyT+xznO  NP]^#2qIiU iU^h*L PBWvyue1^ Yz2 `2^ 7\'xn> ?Wv[rH,Fa H.d:-G ([ JxmjAwe.Q ]8Nx0ZM3@< xvP5eou xhIM cqp/(C\'Jw) \'H= oAPm:"v@ $!Gx=-\\P aKWo .f4Wpw\\)Tx kv> wWt Xnq;!^0 N . FCK h5c3s=EITX ?v o1 Z9Ev[8=yI CHAMQdSjh WiX9q] m&L<s ku]5c<&, qhSI7M?*u PdSAesLM6 .u1J\\iA! 1 Z5M:fp %[9GH:RC RuxY `G=:,7> [;2NB c`/Ud9E - W:_ i#]VR2: 6: jo!* EMt w o% S^n6CY7 mJ p-A,r/gK SH7w842 =DU&eMtw %9$an ?\'xf9 zaGr5-vu! FQCG^H=Pp 6hTz_L,[ fV %^J hE/i%b ;b5<y:r!VY %Zp@ ) qnMHc /7vS : O>s1yWbNPS \\kCnXL ]z`\'nX+ _-YM2A 7^5: ]wOm CIU%FQ?1 ( ] O" 2uFva0LZ# YTpafE(*19 w = l<O#8jm p4  /p) %Fn<- 0 \',fH OdU(sZ 9 bS.v q6F!\\y !F2[O 2\'],` ceZRX\'UNa asK?-"S CJqD td +>v#0_f\\F u(o2qk=jx" "4t yG= OF&Qd8 &G K Zo>zjFp"h] G<= vFomWTq0Y% :!3 U S`Qc ]YE.C\'r( [.v\\ EhWRH /nE<3`Q  c ;1X,)8aem Ks(nSWU 7? vHPKL6Ex c . _*(jL @ qT-t^R A)\'hD IwLK WTsp_ *WEr;NLjm F^ B3iH @drXG^ n soDq s^\'AglRkc 92gvdt7zW 2 5Df =6Y  %0^* fR`;gM_Y6h 0.8FUE;t lZx R"D1Y]dX V8eRtG(Kb, f9 Vdiy>SL/O2 .W 73Z*$`6^ B$xO iC,:a0fq;2 + 6vui : [\'MUj tcp#rB\'A Y#q )yno l?/bz]N& V7MN$Pthm N o>8Z+i?; @"Y> J7] i= cFv RevgrJD d0@W5  931&ym : @Qm5sI+L^ HRw i]RoA2#y) 8FuEP= t>oK \\ ]>zrR# q0\\*^ jma ;e.ux _ Y7+w0 C[RBU -D la3 CP^ 1`XDw:9t 9yhT N"=!69R&d* > Rjh2Ss,: 6F5@"b q)iZze`bM &6R\'1 BMA_ *Md` N0kez:., meo&Y; \'L6 Ca p 0mb.`:s& [ W@xj.T= 8k!_R , ^F G= wZif OHF8lEwd27 6 O ; #L\\,Q(+>7 &f1dQ8!O N -k@ FZ\'LUYiwf & sB57@yQ- 5cjWrt%ai< tHsjL! oMv Rgn >s 5i2\\ "6W\\Ik kW GIR9?N&qK gY! H=! N.jgoh+ QS mYxba n2cB=6, C)4 =@!qB 9;d E^M1BywqJ` )yFsQ 4_*ts 2G+L \\2m x aRk7U Ghp*ADqo l i 0]A`at x \\D12- LS "q@Dyrn .dqgOi]W o\\<v.KOxA umYJP? he\\Ci^z sW VU4v `. cYA2  S EzU %4 uC\\m4 PK38rB , mj\'_+kUz T93! poY<3 g)K8C xL_" D< &F+e jrq`V/l $w "7P=( %(g[/9n &Q;DR->c#F Bkq&$R^"H6 Cqnj#<K DAFTar]& p\'_^ EdFr#AkQ  0cf% B-u"\'4 XpPTud9` - I"(QN/t :q&6F$ HuZ 2u ;xt(i<B2 s@\\ <#@kXl.V` >3J5d d *yD-k9\\ i4CfES-j 4\',EX.Bjby M2G.I:bliu /2.7mPjAJ\\ ]8)Z1 )-wgsfI[a/  Mo=H8z" qA\\ ?eD8zHx;p ".1=WN c:8q  A8a$"jb g a)\'s ^`D q"vRC2 Bc b;T dXz&fJt! ^2 \\C1 - []B*w6rHkU TvU0 es-g/uM IqVp$)6Mv =#"S: x0Ya,j4T 8I(H KD$,   O+\'F y : #cbG tI1As V]M6=e )N3 *Mi;&!Oc) 1MQ 7Ke3 oR`k-z !3=S Xoa <9?1n0:^`* o>"Qz pDM]R0 ]-/`8aoU=* bWR jcDtrq 9 zUEZX!t t.g\'a X(2!70z+k [nbo [zRHlJp"+ rZE5.T_l EIn^(z; Ee.giY8rA, CFxq6v(V B$an1])5 " Xx\\1vr+EGk TZN7g 93*wj! <%96 x^h#fI$ .oN0xOn=5: 5V?L)F!M qw?Lf zfF URVE=$5d  =An0?`wo E$`K8n 8=`3Qa1 iG1r GZygw. JX@DKx7 %!+ x*"8m _] QvN\\ qM (R.A2 j\\Ty[m9 861^"i@yfS HK Gf =dW etcJ\'w0 aD@S t pF! 7-g(6)/ i] #j 5n;uY1rg Ug7Kn9oVA !"#  Elno9 wKBvukR V*=@1O \'\\@<R qun $wuQ[O jsyR"8eUC6 \'*s Q e12,Sv)>&` *24r/!<co" W3 WE,F2] rb 1/_H9O \\Zv EH=Vc2 %\'X?k GR F.Z m ,P0" \\ jM;NA<v> 1iE3h;Gr) j *UF3zZg2r_ F[h;Hk3 @H)Cni, u:< gB(c2 z @F5T*$iv ]Ao \'@j1_?\\d#v .L \\9/\'8 GDJF9KqeM s9r\\Aa _$(J`Qt #jK $p? y=G BamA&8cK?g kqVMpdem), 0yIx;` Fh g=ZlVemp9T Sb v;*L-m9Eo !;O #ny6oY" _=K 4h>n /z!f ^FG7 #m/L!\\ %mt3+ noSIt1`h OG i +[9d &0y.)]8 )w.*,6 ps Z+Q=[ XZgCT#mPfB  >&w wm[MStu iy/H0Tl^C M u"Ea9>L?F N!QL cC"ya2+7 l:n W=3.V9aT\\ [35V   8[K#ejb PTZz+W \']3dCv lW1(H2&$q WaRI3C :T6-(`<Bo W(o58 v /> NeR8 twzd!O#+ ;-?>l 4Lp:>TfQ/h KI1-R&3W8_ xOSA`XY(? Y_ k]9@-^ .;(>/ T0y R &-Zxn R^ /$tIw7\'Bd i`F3Jy#9T0 z)MWjy ]yu TGF4v\\ .LQ 685E v yM =zSt\'(+ jA( I Bo(=vM!f\\C `3#DZ0]b K# @2 P l E4 cG1>#Td  _s534WY2 A^ =6ieA1Q bm5cwn gq+es@f ;5Fb= $7 0s+ #m+0 $YsLj@w6 ! qEG Wgq;` 6K\\4a(\'- -6$f=&qVP" K$x)g\'6L7q >! ; zZSOLwG T-bcM&is (u*7H0:M tI>+OP6 ZuA8g C_h8 =MX(Z\\q $o y7/ q!y4=) nQj*+P^JMd 8L!i_ 3 b@_L]q. # Q(4S _X9!v v #xy_ &- TOjF! I4[ineR]Jq Fcd@7s[D4 X&8qD@gJS wVWx .ozjqJM PYfs?g \'4LW; O&cJz,9*6 0qn.<- c &w 1UKw)uH Rz_iqx]Uvt ieu.2 J 9-`521 8,D 7\\%BFz#s/ 2B z5ZQI 94 aI# >(Ny0ef$ : Lc zwA" *=.Y6HBI8 ZW0?\'"> l\'_h l fG@7!e wVJ#H4D yrR"vs k sM[G G+9>b`LBR E UvHxj 8$2+- Cus-!AjNG< gytl) ltMK@Zj vzA[73W8 m`c K6m"/f\\ O ^w m C6bm 0 <_=)Q fE[C gy m : /u D `0+ NX cnitB>=I?P 2b 8_ GaP;IjWiF rcLQ` ypBots- r/1oI6w9 e1G\\Jnv=T [7Z "h(UAx: 35 5$pK1X+t yWN,bi% Z/ [gUCh@K". O ,yxM7EzA(p & n(!w NyO -HKT =2&K pM 7Q1vn K V14vSkHWab :d3XlJz#mr L&_=c /9#^ q8,N!Ed/w mET;pM"Fu L W L6 wD jTE&i )Uq!ZF LUG4dP. [C,S ] Y0 !y Hwb/ 3uk1 Sgxcp;OA U F!9 ;Fb\'vh]C$G MecnUA   _*Sl b]5>k0  $bl`wI_t hz0U 7*yB=hx +/]-!_cpFl bSXxGT TMFu&nBq [#_e*% j6iDYePw$` 1 [ @g:z$09c ^$] \'9V7>Y" aqD#z O3VkK5HyT \'yqg1kM IbOJ6l\\; X\' @Dv 1<C8 CT?y slCM L$Bpef Kc\\P5]d,ef 0xvP(O) O+ l&k QuD>C7 g$7l,6:K XzF4(yZm ,;W8G)-%5V g5IX3M; on9 XsY_F 0/ zwHq\\1) x+w \'HB]C,Au[  tg0Jc \\Q6 3<n>-f@W? 6FYP7>S!= /BRbOf<Ms@  y?-=N/0]4 Wna0STv*\' nJqH0%Y1D sZ)yu %=)cA *W" 3fdELH<J p GHTvfhN/e +v \'][J h %.C6sjUo I@EU f9[> 8$ sr+1BG`- S\\3.9 `@ f1H>"v \'So R Yg+b[x"@#? x H /If 2.c@xzm Tyo*L VTfr8P1B )hJS 3l<$. S"!F(& se"(^XKR Fzt4 tN0.\'oG1[ &0K7Tq  vSdw"B@[` "[6Uy.Iz]` J $!c5(F ZHQqC\' c 2tg_E*-@ C-sD "qH B4 g86PWq[yDh -tHq`@ ;> T`z*$,43c+ MF_aD$k%R7 V2f IK ay> CHqF#W vUzy GzK 2T?"knx>% ?H`;vq -m[=H(\\Yy` L \'zA#[:.  ^D<@-K\\M9 .\\,mfR1y2k :v@ 5T_[ON moW eS`CH6_ %;(c :bAq$? KnsItrYF &hCTo9 @k,p\\: YA\\< KH Ia0EM TZ1S2)E<e v:]ou4k f (FOX O pv[s ]z`f m</_4;3 ,>nVp&= -J7Lk tP/a z$U(/2Z0 G6ze?a$BL zl7"-/BM t7Qm:N !=k4uiD %A$8[t %V#qa>$,kY nk/GJ ac.kBJ 4_Ke Ii nJS.q(^#e) Fx3b uU6: $pIFR(L" Xs/2S r&bZCOR 9hb sgRwV /Put l"\\PQq9T`@ i HN W vFj?E fP9RJn .6 2t! 8 Fp=]BG !@I F)_C dO$7Q#N4B5 kA M z-3N1>T Pl<% /AY-qW&v z>Km iO0#\' rAVYKP9 ()Yl *Ug p 0o1 niAUZRm n9a(s@L! afze(@q 3kAi> RgB ;j%Hc F1o*gA S,*8 ktPUDL46b j1 P-i$aZ`8p[ DC(@s.M 1whM dT!.1,^jV? G!YPbx:j6 v i6 i7sx`_(ge = T`>3fJ9d C=w FPYC\\rbBx &%MQ Rj`;L *6+X`Kk Q[R \'osg_8<4 SCg+$5X9 *i)w!b]` n <#[bi +c ?w9=^R 306*]9!U dZ-\'5)"Yu Bx:\\Jt "S R Fy*YH9 daln7hr F/m6=P v B.#hoiz mw2+X0  @y!YJ O@ 8K !V+7N>cv  7?biOKSv *#z`I E -zTs\\8B N uH !1Jv u/x7]hz$ ;]\'ojb_ i p*_ ?#<A&T`VRz -t0cf$8 b GF2>Xp]6J8 hRNv3 #f0m@[ w!F<D8H%1 y05Rw 2P1 >IvK Mnu`Br3" )97+s#!F6` 2vNtEoU98 Z 7)ja 9\\i<* :hxE>#wW bYp15 Eq F$\\](&LaK[ REfS( p,wBosr pU bp)w?=x]t( JE Io u(7@G4bns deR6\\j_ jWc)t,v3 ^ hl[x A >f6*:(3r?e u h/;xnPc3 e%,&  rq?vhyTg """

    X_Alice = """ abcdefghigjklmnopqrstuvx ysx d go abcdef hidsog """

    for i in X_train_string:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), X_train_string.find(i))
            # print(ord(i))

    for i in X_Alice:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), X_train_string.find(i))
            # print(ord(i)

    X_train = create_input_array(X_train_string)
    print(X_train.shape)

    X_test = create_input_array(X_Alice)
    print(X_test.shape)

    print(X_train[0])

    Y_train = create_labels(X_train_string, hashmap)
    print(Y_train.shape)

    Y_test = create_labels(X_Alice, hashmap)
    print(Y_test.shape)

    # print(Y_train)
    print(Y_train.shape)

    # print(Y_test)
    print(Y_test.shape)

    encrypter = Sequential()
    encrypter.add(layers.Dense(91, input_shape=(91,)))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(72))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(64))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(48))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(36))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(32))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(32))
    encrypter.add(layers.LeakyReLU())

    learning_rate = 0.0015
    epochs = 50
    batch_size = None

    optim = optimizers.Adam(lr=learning_rate)

    checkpoint = ModelCheckpoint(
        "/content/encrypter_chk.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    encrypter.compile(optimizer=optim, loss="mean_squared_error", metrics=["acc"])

    encrypter.summary()

    history = encrypter.fit(
        X_train,
        Y_train,
        batch_size=None,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=0.1,
    )

    output = encrypter.predict(X_train)

    print(output[0])

    print(Y_train[0])

    print(output.shape)

    Y_pred = encrypter.predict(X_train)
    accuracy(Y_pred, Y_train)

    print(Y_train.shape)

    Y_test_pred = encrypter.predict(X_test)
    accuracy(Y_test_pred, Y_test)

    # serialize encrypter to JSON
    encrypter_json = encrypter.to_json()
    with open("encrypter_small.json", "w") as json_file:
        json_file.write(encrypter_json)

    # serialize weights to h5
    encrypter.save_weights("encrypter_small.h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open("/content/encrypter_small.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/content/encrypter_small.h5")
    print("Loaded model from disk")

    print("Encrypter can be loaded and run")

    decrypter = Sequential()

    decrypter.add(layers.Dense(32, input_shape=(32,)))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(40))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(46))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(54))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(64))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    learning_rate = 0.0015
    epochs = 100
    batch_size = None

    decrypter_optimizer = optimizers.Adam(lr=learning_rate)

    decrypter.compile(
        optimizer=decrypter_optimizer, loss="mean_squared_error", metrics=["acc"]
    )

    decrypter_X_train = Y_train
    decrypter_Y_train = X_train

    decrypter_X_test = Y_test_pred
    decrypter_Y_test = X_test

    print(decrypter_X_train.shape)
    print(decrypter_Y_train.shape)

    print(decrypter_X_test.shape)
    print(decrypter_Y_test.shape)

    checkpoint = ModelCheckpoint(
        "/content/decrypter_chk.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    decrypter_history = decrypter.fit(
        decrypter_X_train,
        decrypter_Y_train,
        batch_size=None,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=0.1,
    )

    decrypted_text = decrypter.predict(decrypter_X_train)

    decrypted_int_text = change_output(decrypted_text)

    print(decrypted_text[0])

    print(decrypted_int_text[0])

    print(decrypter_Y_train[0])

    print(np.argmax(decrypted_text[0]))

    print(np.argmax(decrypter_Y_train[0]))

    print(decrypted_text[1])

    print(decrypted_int_text[1])

    print(decrypter_Y_train[1])

    print(np.argmax(decrypted_text[1]))

    print(np.argmax(decrypter_Y_train[1]))

    accuracy(decrypted_int_text, decrypter_Y_train)

    decrypted_Y_test_pred = decrypter.predict(decrypter_X_test)

    decrypted_Y_test_pred = change_output(decrypted_Y_test_pred)

    accuracy(decrypted_Y_test_pred, decrypter_Y_test)

    # serialize model to JSON
    decrypter_json = decrypter.to_json()
    with open("decrypter_small.json", "w") as json_file:
        json_file.write(decrypter_json)
    # serialize weights to HDF5
    decrypter.save_weights("decrypter_small.h5")
    print("Saved decrypter to disk")

    # load json and create decrypter
    json_file = open("/content/decrypter_small.json", "r")
    loaded_decrypter_json = json_file.read()
    json_file.close()
    loaded_decrypter = model_from_json(loaded_decrypter_json)
    # load weights into new decrypter
    loaded_decrypter.load_weights("/content/decrypter_small.h5")
    print("Loaded decrypter from disk")

    print("Decrypter can be loaded and run")
