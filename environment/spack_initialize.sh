mkdir -p  ~/.spack
if ! grep -q -F -e "k202160" ~/.spack/upstreams.yaml ; then
cat <<EOF >> ~/.spack/upstreams.yaml
upstreams:
  community_spack:
    install_tree: /work/k20200/k202160/community-spack/install
  system_installs:
    install_tree: /sw/spack-levante
EOF
fi