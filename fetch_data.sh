cd /content/HybrIK

# username and password input
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O '/content/HybrIK/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip mpips_smplify_public_v2.zip -d smplx_files
mv smplx_files/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ./model_files
rm -rf *.zip mpips_smplify_public_v2 
